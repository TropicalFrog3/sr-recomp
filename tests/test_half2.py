"""sm86/translate/glsl/{alu,thread,head,core}.py: FP16x2 immediate lane order (H0 in S{slot}1, H1 in S{slot}), LDSM
subgroup barriers, and HMMA VK_KHR_cooperative_matrix (<key>.cm.glsl) emission, checked without the game or a GPU.

Why: AluOps.half2 read S{slot} as H0 and S{slot}1 as H1 (swapped), so Pass 0104's asymmetric
HMUL2 R5, R5, 2.3984375, 0.07745361328125 (Sb=0x40cc H1, Sb1=0x2cf5 H0) applied the 2.4 exponent scale to the linear
lane and 1/12.92 to the log2 lane; ThreadOps.ldsm adds subgroupBarrier() around ldsm() to order shared-memory
accesses even without BAR.SYNC; HMMA compute shaders emit b.reflection["cm_glsl"] with specialization constants
1024..1039 and coopMatMulAdd.

Usage: uv run pytest tests
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sm86/translate"))
sys.path.insert(0, str(ROOT / "sm86/decode"))

import glsl  # noqa: E402
import sass_decode  # noqa: E402
from program import Decoder, Insn  # noqa: E402


def _iv(value: int, sym: str = "", width: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        value=value,
        symbols=[sym] if sym else [],
        slot_values=[(SimpleNamespace(width=width), value)],
    )


def _insn(pc: int, op: str, cls_name: str, fields: dict[str, SimpleNamespace]) -> Insn:
    d = SimpleNamespace(mnemonic=op, cls=SimpleNamespace(name=cls_name))
    return Insn(pc, 0, d, fields)


def test_half2_asymmetric_immediates() -> None:
    # S{slot} holds H1 (bits 63..48) and S{slot}1 holds H0 (bits 47..32); .x must read S{slot}1 and .y S{slot}.
    base = {"Pg": _iv(7), "Pg@not": _iv(0), "iswzA": _iv(0, "H1_H0"), "ofmt": _iv(0, "F16_V2")}
    insns = [
        _insn(
            0x00,
            "HMUL2",
            "hmul2__RI",
            {**base, "Rd": _iv(5), "Ra": _iv(5), "Sb": _iv(0x40CC), "Sb1": _iv(0x2CF5), "sat": _iv(0, "nosat")},
        ),
        _insn(
            0x10,
            "HADD2",
            "hadd2__RI",
            {**base, "Rd": _iv(4), "Ra": _iv(3), "Sc": _iv(0x4000), "Sc1": _iv(0x3C00), "sat": _iv(0, "nosat")},
        ),
        _insn(
            0x20,
            "HFMA2",
            "hfma2__RRI",
            {
                **base,
                "Rd": _iv(6),
                "Ra": _iv(2),
                "Rb": _iv(3),
                "iswzC_as_B": _iv(0, "H1_H0"),
                "Sc": _iv(0x4200),
                "Sc1": _iv(0x3800),
                "satrelu": _iv(0, "nosat"),
            },
        ),
        _insn(
            0x30,
            "HFMA2",
            "hfma2__RIR",
            {
                **base,
                "Rd": _iv(8),
                "Ra": _iv(2),
                "Sb": _iv(0x4400),
                "Sb1": _iv(0x3400),
                "Rc": _iv(3),
                "iswzC": _iv(0, "H1_H0"),
                "satrelu": _iv(0, "nosat"),
            },
        ),
        _insn(
            0x40,
            "HSETP2",
            "hsetp2__RI",
            {
                "Pg": _iv(7),
                "Pg@not": _iv(0),
                "Pu": _iv(2),
                "Pv": _iv(3),
                "Pp": _iv(7),
                "Pp@not": _iv(0),
                "Ra": _iv(4),
                "iswzA": _iv(0, "H1_H0"),
                "Sc": _iv(0x4500),
                "Sc1": _iv(0x292D),
                "cmp": _iv(12, "GEU"),
                "bop": _iv(0, "AND"),
                "ftz": _iv(0, "noftz"),
                "h_and": _iv(0, "noh_and"),
            },
        ),
        _insn(
            0x50,
            "HMNMX2",
            "hmnmx2__RIR",
            {
                **base,
                "Rd": _iv(9),
                "Ra": _iv(1),
                "Sb": _iv(0x4600),
                "Sb1": _iv(0x3000),
                "Pp": _iv(7),
                "Pp@not": _iv(1),
                "nan": _iv(0, "nonan"),
                "xorsign": _iv(0, "noxorsign"),
            },
        ),
        _insn(0x60, "EXIT", "exit_", {"Pg": _iv(7), "Pg@not": _iv(0)}),
    ]
    src, _ = glsl.translate(None, insns, "cs")
    assert "vec2(unpackHalf2x16(0x2cf5u).x, unpackHalf2x16(0x40ccu).x)" in src
    assert "vec2(unpackHalf2x16(0x3c00u).x, unpackHalf2x16(0x4000u).x)" in src
    assert "vec2(unpackHalf2x16(0x3800u).x, unpackHalf2x16(0x4200u).x)" in src
    assert "vec2(unpackHalf2x16(0x3400u).x, unpackHalf2x16(0x4400u).x)" in src
    assert "vec2(unpackHalf2x16(0x292du).x, unpackHalf2x16(0x4500u).x)" in src
    assert "vec2(unpackHalf2x16(0x3000u).x, unpackHalf2x16(0x4600u).x)" in src


def test_ldsm_subgroup_barriers() -> None:
    insns = [
        _insn(
            0x00,
            "LDSM",
            "ldsm__sImmOffset",
            {
                "Pg": _iv(7),
                "Pg@not": _iv(0),
                "sz": _iv(0, "16"),
                "mode": _iv(1, "MT88"),
                "num": _iv(2, "4"),
                "Rd": _iv(28),
                "Ra": _iv(60),
                "Ra_offset": _iv(0x40, width=24),
            },
        ),
        _insn(0x10, "EXIT", "exit_", {"Pg": _iv(7), "Pg@not": _iv(0)}),
    ]
    src, _ = glsl.translate(None, insns, "cs")
    assert "#extension GL_KHR_shader_subgroup_basic : require" in src
    ldsm_line = next(ln for ln in src.splitlines() if "= ldsm(" in ln)
    assert ldsm_line.strip().startswith("subgroupBarrier();")
    assert ldsm_line.strip().endswith("subgroupBarrier();")


def test_hmma_cm_glsl() -> None:
    insns = [
        _insn(
            0x00,
            "HMMA",
            "hmma_16816_f16_f16",
            {
                "Pg": _iv(7),
                "Pg@not": _iv(0),
                "size": _iv(0, "16816"),
                "dstfmt": _iv(0, "F16"),
                "srcfmt": _iv(0, "F16"),
                "Rd": _iv(4),
                "Ra": _iv(8),
                "Rb": _iv(12),
                "Rc": _iv(16),
            },
        ),
        _insn(0x10, "EXIT", "exit_", {"Pg": _iv(7), "Pg@not": _iv(0)}),
    ]
    src, refl = glsl.translate(None, insns, "cs")
    assert "#extension GL_KHR_cooperative_matrix : require" not in src
    assert "cm_glsl" in refl
    cm = refl["cm_glsl"]
    assert "#extension GL_KHR_cooperative_matrix : require" in cm
    assert "#extension GL_KHR_memory_scope_semantics : require" in cm
    assert "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require" in cm
    assert "layout(constant_id = 1024) const int MMA_A0 = 0;" in cm
    assert "layout(constant_id = 1039) const int MMA_C3 = 3;" in cm
    assert "coopMatMulAdd(ma, mb, mc)" in cm
    assert "hmma(r8, r9, r10, r11, r12, r13, r16, r17, hd0, hd1);" in cm


def _encode_class(spec, cls_name: str, overrides: dict[str, int]) -> int:
    cls = next(c for c in spec.classes if c.name == cls_name)
    word = 0
    for item in cls.items:
        val = None
        if item.name in overrides:
            val = overrides[item.name]
        elif item.kind == "constant":
            val = int(item.value)
        elif item.allowed:
            val = min(item.allowed)
        if val is not None and len(item.slots) == 1 and item.slots[0].width > 0:
            word = item.slots[0].deposit(word, val)
    return word


def test_decoded_half2_asymmetric_immediates() -> None:
    spec_path = ROOT / "build/sm86/instructions.json"
    if not spec_path.is_file():
        pytest.skip("no sm86 instructions.json")
    dec = Decoder(spec_path)
    cases = [
        ("hmul2__RI", {"Pg": 7, "Rd": 5, "Ra": 5, "Sb": 0x40CC, "Sb1": 0x2CF5, "iswzA": 0}, "0x2cf5u", "0x40ccu"),
        ("hadd2__RI", {"Pg": 7, "Rd": 4, "Ra": 3, "Sc": 0x4000, "Sc1": 0x3C00, "iswzA": 0}, "0x3c00u", "0x4000u"),
        (
            "hfma2__RRI",
            {"Pg": 7, "Rd": 6, "Ra": 2, "Rb": 3, "Sc": 0x4200, "Sc1": 0x3800, "iswzA": 0, "iswzC_as_B": 0},
            "0x3800u",
            "0x4200u",
        ),
        (
            "hfma2__RIR",
            {"Pg": 7, "Rd": 8, "Ra": 2, "Rc": 3, "Sb": 0x4400, "Sb1": 0x3400, "iswzA": 0, "iswzC": 0},
            "0x3400u",
            "0x4400u",
        ),
        (
            "hsetp2__RI",
            {"Pg": 7, "Pu": 2, "Pv": 3, "Pp": 7, "Ra": 4, "Sc": 0x4500, "Sc1": 0x292D, "cmp": 1, "bop": 0, "iswzA": 0},
            "0x292du",
            "0x4500u",
        ),
        (
            "hmnmx2__RIR",
            {"Pg": 7, "Rd": 9, "Ra": 1, "Sb": 0x4600, "Sb1": 0x3000, "Pp": 7, "iswzA": 0},
            "0x3000u",
            "0x4600u",
        ),
    ]
    words = b""
    for cls_name, ov, _, _ in cases:
        w = _encode_class(dec.spec, cls_name, ov)
        d = sass_decode.decode(dec.spec, w)
        assert d.status == "ok" and d.cls.name == cls_name
        words += w.to_bytes(16, "little")
    _, insns = dec.program(words, "cs")
    src, _ = glsl.translate(None, insns, "cs")
    for _, _, h0, h1 in cases:
        assert f"vec2(unpackHalf2x16({h0}).x, unpackHalf2x16({h1}).x)" in src

