#!/usr/bin/env python3
"""
Script to clean up conda base environment by removing non-essential packages.
This preserves core conda functionality while removing project-specific packages.
"""

import subprocess
import sys

# Core packages that should NEVER be removed from base environment
ESSENTIAL_PACKAGES = {
    # Core conda/python packages
    'conda', 'conda-anaconda-telemetry', 'conda-anaconda-tos', 'conda-content-trust',
    'conda-libmamba-solver', 'conda-package-handling', 'conda_package_streaming',
    'libmambapy', 'menuinst', 'anaconda-anon-usage',
    
    # Core Python packages
    'pip', 'setuptools', 'wheel', 'packaging', 'typing_extensions',
    
    # Essential system packages
    'certifi', 'charset-normalizer', 'cryptography', 'idna', 'urllib3', 'requests',
    'pysocks', 'truststore', 'platformdirs', 'distro',
    
    # Core data structures and utilities
    'zstandard', 'pycosat', 'ruamel.yaml', 'ruamel.yaml.clib', 'boltons',
    'jsonpatch', 'jsonpointer', 'archspec', 'frozendict',
    
    # Essential Python tools
    'six', 'python-dateutil', 'pytz', 'tzdata', 'filelock',
    
    # Core C/system dependencies
    'cffi', 'pycparser', 'brotli', 'lxml', 'chardet'
}

# Packages installed for the DeepSeek project that should be removed from base
PROJECT_PACKAGES = {
    # ML/AI packages
    'torch', 'torchvision', 'torchaudio', 'transformers', 'accelerate',
    'datasets', 'evaluate', 'deepspeed', 'wandb', 'trl', 'peft', 'bitsandbytes',
    'huggingface-hub', 'tokenizers', 'safetensors',
    
    # Data science packages
    'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'scipy',
    'bottleneck', 'numexpr', 'joblib', 'threadpoolctl',
    
    # Jupyter and related
    'jupyter', 'notebook', 'jupyterlab', 'ipython', 'ipykernel', 'ipywidgets',
    'jupyter-client', 'jupyter-console', 'jupyter-core', 'jupyter-events',
    'jupyter-lsp', 'jupyter_server', 'jupyter_server_terminals', 'jupyterlab_pygments',
    'jupyterlab_server', 'jupyterlab_widgets', 'notebook_shim', 'nbconvert',
    'nbformat', 'nbclient', 'widgetsnbextension',
    
    # Visualization and plotting
    'contourpy', 'cycler', 'fonttools', 'kiwisolver', 'pyparsing', 'pillow',
    'matplotlib-inline',
    
    # Development tools
    'debugpy', 'ipython_pygments_lexers', 'jedi', 'parso', 'pexpect', 'ptyprocess',
    'prompt_toolkit', 'pygments', 'decorator', 'executing', 'pure-eval',
    'stack-data', 'asttokens', 'traitlets', 'wcwidth',
    
    # Web/HTTP related
    'aiohttp', 'aiohappyeyeballs', 'aiosignal', 'anyio', 'attrs', 'click',
    'frozenlist', 'fsspec', 'h11', 'httpcore', 'httpx', 'multidict', 'propcache',
    'sniffio', 'yarl', 'websocket-client', 'tornado', 'terminado',
    
    # Additional utilities
    'appnope', 'arrow', 'async-lru', 'babel', 'beautifulsoup4', 'bleach',
    'comm', 'defusedxml', 'dill', 'einops', 'fastjsonschema', 'fqdn',
    'gitdb', 'gitpython', 'hf-xet', 'hjson', 'isoduration', 'jinja2',
    'json5', 'jsonschema', 'jsonschema-specifications', 'markdown-it-py',
    'markupsafe', 'mdurl', 'mistune', 'msgpack', 'multiprocess', 'narwhals',
    'nest-asyncio', 'networkx', 'ninja', 'overrides', 'pandocfilters',
    'prometheus_client', 'protobuf', 'psutil', 'py-cpuinfo', 'pyarrow',
    'pydantic', 'pydantic_core', 'pyyaml', 'pyzmq', 'referencing', 'regex',
    'rfc3339-validator', 'rfc3986-validator', 'rich', 'rpds-py', 'send2trash',
    'sentry-sdk', 'setproctitle', 'smmap', 'soupsieve', 'sympy', 'tinycss2',
    'tqdm', 'uri-template', 'webcolors', 'webencodings', 'xxhash',
    'types-python-dateutil', 'argon2-cffi', 'argon2-cffi-bindings',
    'python-json-logger', 'pluggy'
}

def get_installed_packages():
    """Get list of currently installed packages."""
    try:
        result = subprocess.run(['pip', 'list', '--format=freeze'], 
                              capture_output=True, text=True, check=True)
        packages = []
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                package_name = line.split('==')[0].lower()
                packages.append(package_name)
        return packages
    except subprocess.CalledProcessError as e:
        print(f"Error getting package list: {e}")
        return []

def main():
    print("🧹 Cleaning up conda base environment...")
    print("This will remove ML/data science packages while preserving core functionality.\n")
    
    # Get currently installed packages
    installed = get_installed_packages()
    
    # Find packages to remove (intersection of installed and project packages)
    to_remove = []
    for pkg in installed:
        pkg_lower = pkg.lower().replace('_', '-')
        if pkg_lower in PROJECT_PACKAGES and pkg_lower not in ESSENTIAL_PACKAGES:
            to_remove.append(pkg)
    
    if not to_remove:
        print("✅ No project packages found in base environment to remove.")
        return
    
    print(f"📦 Found {len(to_remove)} packages to remove:")
    for pkg in sorted(to_remove)[:10]:  # Show first 10
        print(f"  - {pkg}")
    if len(to_remove) > 10:
        print(f"  ... and {len(to_remove) - 10} more")
    
    print(f"\n⚠️  Essential packages will be preserved:")
    print(f"  - conda, pip, setuptools, wheel, certifi, etc.")
    
    print(f"\n🚀 Proceeding with cleanup...")
    # Auto-proceed for non-interactive environments
    
    # Remove packages in batches to avoid command line length limits
    batch_size = 20
    for i in range(0, len(to_remove), batch_size):
        batch = to_remove[i:i+batch_size]
        print(f"\n🗑️  Removing batch {i//batch_size + 1}/{(len(to_remove) + batch_size - 1)//batch_size}...")
        
        try:
            subprocess.run(['pip', 'uninstall'] + batch + ['-y'], 
                          check=True, capture_output=True)
            print(f"✅ Successfully removed {len(batch)} packages")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error removing batch: {e}")
            print("Continuing with next batch...")
    
    print(f"\n🎉 Base environment cleanup complete!")
    print("💡 Remember to install packages in your project environment:")
    print("   conda activate deepseek_verilog")
    print("   pip install -r requirements_mac.txt")

if __name__ == "__main__":
    main()