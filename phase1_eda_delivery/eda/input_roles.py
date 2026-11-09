"""E02: fail-closed role/name verification for official Amazon Reviews'23 inputs.

A filename check is structural evidence, NOT proof of authentic source bytes.
Noncanonical names are allowed only by an explicit, auditable override.
"""
from pathlib import Path

CATEGORIES = ("All_Beauty", "Appliances", "Beauty_and_Personal_Care", "Electronics")
ROLES = ("reviews", "metadata")


def _without_extension(path):
    name = Path(path).name
    if name.endswith(".jsonl.gz"):
        return name[:-9]
    if name.endswith(".jsonl"):
        return name[:-6]
    return name


def validate_dataset_roles(datasets, config):
    """Require official filename/category agreement or explicit noncanonical override.

    This is deliberately separate from reference structure_audit.py, which is immutable.
    Unknown alternative names are not silently certified as the authentic originals.
    """
    if set(datasets) != set(CATEGORIES):
        raise ValueError("E02: category mismatch")
    override = config.get("allow_noncanonical_input_names", False)
    if type(override) is not bool:
        raise ValueError("E02: allow_noncanonical_input_names must be a JSON boolean")
    observed = {}
    for category in CATEGORIES:
        if set(datasets[category]) != set(ROLES):
            raise ValueError(f"E02: missing or unexpected input role for {category}")
        observed[category] = {}
        for role in ROLES:
            base = _without_extension(datasets[category][role])
            expected = category if role == "reviews" else f"meta_{category}"
            if base != expected and not override:
                raise ValueError(
                    f"E02: {category}/{role} has source name {base!r}, expected {expected!r}. "
                    "Check category/role mapping. For deliberately renamed local files, "
                    "set allow_noncanonical_input_names=true and retain hashes/provenance."
                )
            observed[category][role] = {
                "basename": Path(datasets[category][role]).name,
                "expected_stem": expected,
                "matches_official_stem": base == expected,
                "noncanonical_override_used": base != expected,
                "note": "Filename agreement is not independent source authenticity evidence."
            }
    return {
        "requirement": "E02",
        "categories": list(CATEGORIES),
        "role_count": 8,
        "policy": "official filename stem or explicit override",
        "override_configured": override,
        "roles": observed,
        "all_names_match": all(v["matches_official_stem"] for item in observed.values() for v in item.values()),
        "identity_verified_by_source_bytes": False,
    }
