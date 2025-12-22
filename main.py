import importlib

LIBS = {
    "tensorflow": "tf",
    "numpy": "np",
    "pandas": "pd",
    "scipy": None,
    "PIL": None,               # pillow
    "matplotlib.pyplot": "plt",
    "seaborn": "sns",
    "imageio": None,
    "jupyter": None,
    "neptune": None,
}

def gpu_check_tf():
    """Check TensorFlow GPU visibility."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"✔ TensorFlow sees {len(gpus)} GPU(s): {gpus}")
        else:
            print("✘ TensorFlow sees no GPUs")
    except Exception as e:
        print(f"✘ GPU check failed: {e}")

def main():
    print("Import smoke-test starting...")
    for mod_path, alias in LIBS.items():
        try:
            mod = importlib.import_module(mod_path)
            name = alias or mod_path
            version = getattr(mod, "__version__", "n/a")
            print(f"✔ Imported {name} (version {version})")
        except Exception as e:
            print(f"✘ Failed to import {mod_path}: {e}")
    gpu_check_tf()

if __name__ == "__main__":
    main()