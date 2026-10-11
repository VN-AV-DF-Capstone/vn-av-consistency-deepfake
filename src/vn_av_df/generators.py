"""Registry generator: tên dữ liệu phải khớp adapter và checkpoint thực tế."""

MECHANISMS = ("mech_blend", "mech_lowres", "mech_vae", "mech_warp", "mech_temporal")
DIAGNOSTICS = ("diag_paste", "diag_global")


def generator_kind(name):
    """ai: generator thật (worker); mech: MechFake (nhãn AI 1); diag: chẩn đoán E6 (nhãn 0)."""
    if name.startswith("mech_"):
        return "mech"
    if name.startswith("diag_"):
        return "diag"
    return "ai"


def generator_adapter(name):
    """Không tự thay generator khi thiếu dependency hoặc weight."""
    if name == "wav2lip_gan":
        from vn_av_df import wav2lip

        return wav2lip
    if name == "musetalk_1_5":
        from vn_av_df import musetalk

        return musetalk
    if name in MECHANISMS or name in DIAGNOSTICS:
        from vn_av_df import mechfake

        return mechfake.Adapter(name)
    raise ValueError(f"Unsupported generator: {name}")


def setup_generators(cfg):
    """Chuẩn bị generator của phiên (generation_generators) hoặc mọi generator trong plan."""
    mapping = cfg["generation"].get("generators_by_split", {"train": ["wav2lip_gan"]})
    names = cfg.get("generation_generators") or {g for names in mapping.values() for g in names}
    return {name: generator_adapter(name).setup(cfg) for name in sorted(names)}
