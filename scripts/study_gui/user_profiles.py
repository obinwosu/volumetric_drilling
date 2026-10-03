"""Load and save study-session user profiles."""
import pathlib
import re

import yaml

DEFAULT_BURR_MM = 6
VALID_BURR_SIZES = (1, 2, 4, 6)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_").lower()
    return slug or "user"


class UserProfile:
    def __init__(self, name: str, starting_burr_mm: int = DEFAULT_BURR_MM, path=None):
        self.name = name.strip()
        self.starting_burr_mm = starting_burr_mm if starting_burr_mm in VALID_BURR_SIZES else DEFAULT_BURR_MM
        self.path = pathlib.Path(path) if path else None

    def to_dict(self):
        return {
            "name": self.name,
            "starting_burr_mm": self.starting_burr_mm,
        }

    @classmethod
    def from_dict(cls, data, path=None):
        name = str(data.get("name", "")).strip()
        burr = int(data.get("starting_burr_mm", DEFAULT_BURR_MM))
        return cls(name=name, starting_burr_mm=burr, path=path)


class UserProfileStore:
    def __init__(self, profiles_dir=None):
        if profiles_dir is None:
            profiles_dir = pathlib.Path(__file__).resolve().parent / "profiles"
        self.profiles_dir = pathlib.Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self):
        profiles = []
        for path in sorted(self.profiles_dir.glob("*.yaml")):
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
                profile = UserProfile.from_dict(data, path=path)
                if profile.name:
                    profiles.append(profile)
            except Exception as exc:
                print(f"WARNING! Skipping profile {path}: {exc}")
        return profiles

    def create(self, name: str, starting_burr_mm: int = DEFAULT_BURR_MM):
        name = name.strip()
        if not name:
            raise ValueError("Profile name cannot be empty")
        slug = _slugify(name)
        path = self.profiles_dir / f"{slug}.yaml"
        # Avoid clobbering an existing file with a different display name.
        if path.exists():
            n = 2
            while (self.profiles_dir / f"{slug}_{n}.yaml").exists():
                n += 1
            path = self.profiles_dir / f"{slug}_{n}.yaml"
        profile = UserProfile(name=name, starting_burr_mm=starting_burr_mm, path=path)
        self.save(profile)
        return profile

    def save(self, profile: UserProfile):
        if profile.path is None:
            profile.path = self.profiles_dir / f"{_slugify(profile.name)}.yaml"
        profile.path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile.path, "w") as f:
            yaml.safe_dump(profile.to_dict(), f, default_flow_style=False, sort_keys=False)
        return profile
