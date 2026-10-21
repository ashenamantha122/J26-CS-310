"""
LAAX Component 1: Adaptive eBPF Endpoint Agent Packaging.
Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310
"""

from setuptools import find_packages, setup

setup(
    name="laax-agent",
    version="0.2.0",
    description="Adaptive eBPF-Based Lightweight Endpoint Agent for Hybrid Enterprise-IoT XDR",
    author="Amantha M.A",
    author_email="it23184312@my.sliit.lk",
    packages=find_packages(include=["laax", "laax.*", "services", "services.*", "scenario", "scenario.*"]),
    py_modules=["laax_agent", "run_broker", "run_demo", "run_live", "benchmark_overhead"],
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "full": ["networkx>=3.0", "torch>=2.0"],
        "zmq": ["pyzmq>=24", "protobuf>=3.20"],
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "laax-agent = laax_agent:cli_entry",
            "laax-broker = run_broker:cli_entry",
            "laax-benchmark = benchmark_overhead:cli_entry",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
    ],
)
