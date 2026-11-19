// Adapted from MacNativeGames games/totk/runtime/motion_fusion.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// Six-axis sensor states (nn::hid::SixAxisSensorState) from gyro and accelerometer samples. Input per sample, in the
// Switch controller frame (x to the right, y toward the top edge, z out of the face): acceleration in G including
// gravity, as gravity reads (lying face up z = -1 G), and rotation rate in rad/s. Output per sample, as the Switch
// reports it: acceleration in G, angular velocity in revolutions per second, angle = the integrated revolutions per
// axis, direction = the controller's three axes in a reference frame whose z is up. The orientation is integrated from
// the gyro and, while the controller is not being accelerated (|a| within 15 % of 1 G), pulled toward the measured
// gravity (Mahony, gain 0.5/s: about 2 s), as the Switch's sensor fusion does; yaw has no reference and drifts with the
// gyro's bias. The newest MOTION_RING samples are kept; reads are newest first. The feeding thread writes, guest
// threads read, both under the instance's SRW lock.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "input.h"
#include "platform/platform.h"

enum { MOTION_RING = 64 };
#define STALE_NS 200000000ull   // no sample for 200 ms: no motion

struct MotionFusion {
    PlatMutex lock;
    MotionState ring[MOTION_RING];
    uint64_t produced, last_ns;
    double R[3][3];   // controller -> reference; row i = reference axis i in controller coordinates
    double angle[3];
    int restart;
    double bias[3], still_sum[3], still_sq[3];
    int still_count, calibrated, force_cal;
    double sens, deadzone_rad;
    int inv_x, inv_y, auto_cal;
    double stick_dx_rad, stick_dy_rad;
};

MotionFusion *motion_new(void) {
    MotionFusion *m = calloc(1, sizeof *m);
    if (!m) rt_fatal("motion: out of memory");
    plat_mutex_init(&m->lock);
    m->restart = 1;
    m->sens = 1.0;
    m->auto_cal = 1;
    return m;
}

static MotionFusion *global;

static void global_init(void *arg) {
    (void)arg;
    global = motion_new();
}

MotionFusion *motion_global(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, global_init, NULL);
    return global;
}

static double norm3(const double v[3]) { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
// Rows, from the up row (the one the gravity correction keeps right).
static void orthonormalize(double R[3][3]) {
    double n = norm3(R[2]);
    for (int j = 0; j < 3; j++) R[2][j] /= n;
    double d = R[0][0] * R[2][0] + R[0][1] * R[2][1] + R[0][2] * R[2][2];
    for (int j = 0; j < 3; j++) R[0][j] -= d * R[2][j];
    n = norm3(R[0]);
    for (int j = 0; j < 3; j++) R[0][j] /= n;
    cross3(R[2], R[0], R[1]);
}
// A fresh orientation: z up against the measured gravity, x as close to the controller's x as that allows.
static void align_to(double R[3][3], const double acc[3]) {
    double n = norm3(acc), up[3] = {0, 0, 1}, x[3] = {1, 0, 0};
    if (n > 0.5)
        for (int j = 0; j < 3; j++) up[j] = -acc[j] / n;
    if (fabs(up[0]) > 0.9) { x[0] = 0; x[1] = 1; }
    memcpy(R[2], up, sizeof up);
    memcpy(R[0], x, sizeof x);
    orthonormalize(R);
}
// Body-frame rate: R = R * exp([w] dt), the rate plus the gravity pull.
static void integrate(double R[3][3], const double gyro[3], const double acc[3], double dt) {
    double w[3] = {gyro[0], gyro[1], gyro[2]}, n = norm3(acc);
    if (fabs(n - 1) < 0.15) {   // not being accelerated: turn the estimated down toward the measured one
        const double kp = 0.5;
        double measured[3] = {acc[0] / n, acc[1] / n, acc[2] / n}, estimated[3] = {-R[2][0], -R[2][1], -R[2][2]}, e[3];
        cross3(measured, estimated, e);
        for (int j = 0; j < 3; j++) w[j] += kp * e[j];
    }
    double rate = norm3(w), theta = rate * dt;
    if (theta < 1e-12) return;
    double k[3] = {w[0] / rate, w[1] / rate, w[2] / rate}, s = sin(theta), c = 1 - cos(theta);
    double K[3][3] = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}}, E[3][3], out[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kk = 0;
            for (int q = 0; q < 3; q++) kk += K[i][q] * K[q][j];
            E[i][j] = (i == j) + s * K[i][j] + c * kk;
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double v = 0;
            for (int q = 0; q < 3; q++) v += R[i][q] * E[q][j];
            out[i][j] = v;
        }
    memcpy(R, out, sizeof out);
    orthonormalize(R);
}

static void update_stillness_bias(MotionFusion *m, const double acc[3], const double gyro[3]) {
    if (!m->auto_cal && !m->force_cal) {
        m->still_count = 0;
        return;
    }
    double raw_w = norm3(gyro), acc_mag = norm3(acc);
    if (raw_w >= 0.06 || fabs(acc_mag - 1.0) >= 0.10) {
        memset(m->still_sum, 0, sizeof m->still_sum);
        memset(m->still_sq, 0, sizeof m->still_sq);
        m->still_count = 0;
        return;
    }
    for (int i = 0; i < 3; i++) {
        m->still_sum[i] += gyro[i];
        m->still_sq[i] += gyro[i] * gyro[i];
    }
    m->still_count++;
    if (m->still_count < 50) return;
    double inv_n = 1.0 / (double)m->still_count;
    int low_var = 1;
    double mean[3];
    for (int i = 0; i < 3; i++) {
        mean[i] = m->still_sum[i] * inv_n;
        double var = m->still_sq[i] * inv_n - mean[i] * mean[i];
        if (var >= 0.0004) low_var = 0;
    }
    if (low_var) {
        for (int i = 0; i < 3; i++)
            m->bias[i] = (!m->calibrated || m->force_cal) ? mean[i] : (0.8 * m->bias[i] + 0.2 * mean[i]);
        m->calibrated = 1;
        m->force_cal = 0;
    }
    memset(m->still_sum, 0, sizeof m->still_sum);
    memset(m->still_sq, 0, sizeof m->still_sq);
    m->still_count = 0;
}

void motion_feed(MotionFusion *m, const double acc[3], const double gyro[3], uint64_t now) {
    plat_mutex_lock(&m->lock);
    double dt = m->last_ns && now > m->last_ns ? (double)(now - m->last_ns) / 1e9 : 0.004;
    if (dt > 0.05) dt = 0.004;   // a gap (no controller, sensors were off): no integration across it
    m->last_ns = now;
    update_stillness_bias(m, acc, gyro);

    double g[3] = {gyro[0] - m->bias[0], gyro[1] - m->bias[1], gyro[2] - m->bias[2]};
    double w = norm3(g);
    if (m->deadzone_rad > 0.0) {
        double scale = w <= m->deadzone_rad ? 0.0 : (w - m->deadzone_rad) / w;
        for (int i = 0; i < 3; i++) g[i] *= scale;
    }
    for (int i = 0; i < 3; i++) g[i] *= m->sens;
    if (m->inv_y) g[0] = -g[0];
    if (m->inv_x) { g[1] = -g[1]; g[2] = -g[2]; }

    if (m->restart) {
        align_to(m->R, acc);
        memset(m->angle, 0, sizeof m->angle);
        m->stick_dx_rad = m->stick_dy_rad = 0;
        m->restart = 0;
    } else integrate(m->R, g, acc, dt);

    // Accumulate world-up horizontal yaw and body pitch deltas for optional right-stick gyro mode.
    double yaw_rate = -(m->R[2][0] * g[0] + m->R[2][1] * g[1] + m->R[2][2] * g[2]);
    m->stick_dx_rad += yaw_rate * dt;
    m->stick_dy_rad += g[0] * dt;

    MotionState *s = &m->ring[m->produced % MOTION_RING];
    *s = (MotionState){.delta_ns = (int64_t)(dt * 1e9), .sampling = (int64_t)m->produced + 1, .attributes = 1};
    for (int i = 0; i < 3; i++) {
        double rps = g[i] / (2 * M_PI);
        m->angle[i] += rps * dt;
        s->acc[i] = (float)acc[i];
        s->gyro[i] = (float)rps;
        s->angle[i] = (float)m->angle[i];
        for (int j = 0; j < 3; j++) s->dir[i * 3 + j] = (float)m->R[j][i];   // direction i = controller axis i in the reference frame
    }
    m->produced++;
    plat_mutex_unlock(&m->lock);
}

int motion_read(MotionFusion *m, MotionState *out, int max, uint64_t now) {
    if (max <= 0) return 0;
    int n = 0;
    plat_mutex_lock_shared(&m->lock);
    if (m->produced && (now < m->last_ns || now - m->last_ns < STALE_NS)) {
        uint64_t have = m->produced < MOTION_RING ? m->produced : MOTION_RING;
        n = (int)(have < (uint64_t)max ? have : (uint64_t)max);
        for (int i = 0; i < n; i++) out[i] = m->ring[(m->produced - 1 - (uint64_t)i) % MOTION_RING];
    }
    plat_mutex_unlock_shared(&m->lock);
    return n;
}

void motion_restart(MotionFusion *m) {
    plat_mutex_lock(&m->lock);
    m->restart = 1;
    plat_mutex_unlock(&m->lock);
}

void motion_configure(MotionFusion *m, double sens, double deadzone_deg_s, int inv_x, int inv_y, int auto_cal) {
    if (!m) return;
    double cs = sens < 0.0 ? 0.0 : sens > 10.0 ? 10.0 : sens;
    double dz = deadzone_deg_s < 0.0 ? 0.0 : deadzone_deg_s > 45.0 ? 45.0 : deadzone_deg_s;
    plat_mutex_lock(&m->lock);
    m->sens = cs;
    m->deadzone_rad = dz * M_PI / 180.0;
    m->inv_x = inv_x != 0;
    m->inv_y = inv_y != 0;
    m->auto_cal = auto_cal != 0;
    plat_mutex_unlock(&m->lock);
}

void motion_calibrate(MotionFusion *m) {
    if (!m) return;
    plat_mutex_lock(&m->lock);
    memset(m->bias, 0, sizeof m->bias);
    memset(m->still_sum, 0, sizeof m->still_sum);
    memset(m->still_sq, 0, sizeof m->still_sq);
    m->still_count = 0;
    m->calibrated = 0;
    m->force_cal = 1;
    m->restart = 1;
    m->stick_dx_rad = m->stick_dy_rad = 0;
    plat_mutex_unlock(&m->lock);
}

int motion_bias(MotionFusion *m, double bias_rad_s[3]) {
    if (!m) return 0;
    plat_mutex_lock_shared(&m->lock);
    int cal = m->calibrated;
    if (bias_rad_s) memcpy(bias_rad_s, m->bias, sizeof m->bias);
    plat_mutex_unlock_shared(&m->lock);
    return cal;
}

void motion_take_stick_delta(MotionFusion *m, double *dx_rad, double *dy_rad) {
    if (!m) {
        if (dx_rad) *dx_rad = 0;
        if (dy_rad) *dy_rad = 0;
        return;
    }
    plat_mutex_lock(&m->lock);
    if (dx_rad) *dx_rad = m->stick_dx_rad;
    if (dy_rad) *dy_rad = m->stick_dy_rad;
    m->stick_dx_rad = m->stick_dy_rad = 0;
    plat_mutex_unlock(&m->lock);
}
