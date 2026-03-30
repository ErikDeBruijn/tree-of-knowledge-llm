"""Grove adapter registry — tracks validated adapters."""
import json, os, time


class GroveRegistry:
    def __init__(self, registry_dir):
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "registry.json")
        os.makedirs(registry_dir, exist_ok=True)
        if os.path.exists(self.registry_file):
            with open(self.registry_file) as f:
                self.data = json.load(f)
        else:
            self.data = {"registry_version": "0.1.0", "trunk_model": "Qwen/Qwen3-8B", "adapters": []}

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def register(self, adapter_dir, validation_scores=None):
        manifest_path = os.path.join(adapter_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        entry = {
            "name": manifest["name"],
            "contributor": manifest["contributor"],
            "domain": manifest["domain"],
            "path": os.path.abspath(adapter_dir),
            "rank": manifest["architecture"]["rank"],
            "expert_start": manifest["architecture"]["expert_start"],
            "validation_status": "accepted",
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "validation_scores": validation_scores or {},
        }

        # Replace if already exists
        self.data["adapters"] = [a for a in self.data["adapters"] if a["name"] != entry["name"]]
        self.data["adapters"].append(entry)
        self._save()
        return entry

    def list_adapters(self):
        return self.data["adapters"]

    def get_adapter(self, name):
        for a in self.data["adapters"]:
            if a["name"] == name:
                return a
        return None

    def remove(self, name):
        self.data["adapters"] = [a for a in self.data["adapters"] if a["name"] != name]
        self._save()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: registry.py <registry_dir> [list|show <name>]")
        sys.exit(1)
    reg = GroveRegistry(sys.argv[1])
    if len(sys.argv) == 2 or sys.argv[2] == "list":
        for a in reg.list_adapters():
            print(f"  {a['name']} ({a['contributor']}) — {a['domain']} — rank {a['rank']}")
    elif sys.argv[2] == "show" and len(sys.argv) > 3:
        a = reg.get_adapter(sys.argv[3])
        print(json.dumps(a, indent=2) if a else "Not found")
