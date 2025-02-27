import glob
import os
import re
import ast
import subprocess
import sys
from pathlib import Path

import torch
import logging
from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import platform
import urllib.request
import urllib.error
import urllib.parse
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

try:
    import utils.cpp_extension_patch
except:
    logger.info("cpp_extension_patch not found, skipping...")
    pass

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME,
    BuildExtension,
    CUDAExtension,
)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSSING_BUILD_CUDA = os.environ.get("CROSSING_BUILD_CUDA", "1") == "1"
DEVELOP_MODE = "develop" in sys.argv
PACKAGE_NAME = "crossing_cuda_kernels"
BASE_WHEEL_URL = (
    None
)
# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("CROSSING_CUDA_KERNELS_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("CROSSING_CUDA_KERNELS_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("CROSSING_CUDA_KERNELS_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# Compiler flags.
CXX_FLAGS = ["-O3"]
NVCC_FLAGS = ["-O3", "-Wno-deprecated-declarations"]


def _is_hip() -> bool:
    return torch.version.hip is not None


def _is_cuda() -> bool:
    return torch.version.cuda is not None


# warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary in that case.
if _is_hip():
    if ROCM_HOME is None:
        warnings.warn(
            f"ROCM HOME was not found.  Are you sure your environment has nvcc available?  "
            "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
            "only images whose names contain 'devel' will provide nvcc."
        )

    NVCC_FLAGS += ["-DUSE_ROCM"]

if _is_cuda() and CUDA_HOME is None:
    warnings.warn(
        f"CUDA HOME was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def is_nccl_available():
    try:
        nccl_version = torch.cuda.nccl.version()
        return True
    except AttributeError:
        return False


def get_lib_torch():
    torch_dir = Path(torch.__path__[0])
    assert torch_dir.is_dir()
    torch_inc_dir = str(torch_dir / "include")
    torch_lib_dir = str(torch_dir / "lib")
    libraries = [
        "torch",
        "torch_cpu",
        "torch_python",
        "c10",
    ]
    if _is_cuda():
        libraries.extend(
            [
                "torch_cuda",
                "c10_cuda",
                "cudart",
            ]
        )
    if _is_hip():
        libraries.extend(
            [
                "torch_hip",
                "c10_hip",
            ]
        )
    include_dirs = [torch_inc_dir]
    library_dirs = [torch_lib_dir]
    return {
        "libraries": libraries,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
    }


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.2"):
            return nvcc_extra_args + ["--threads", "4"]
        return nvcc_extra_args
    else:
        return []


def append_cuda_extra_compile_flag(extra_compile_args):
    if float(torch.version.cuda) >= 11.0:
        extra_compile_args["cxx"].append("-DENABLE_BF16")
        extra_compile_args["nvcc"].append("-DENABLE_BF16")
    if float(torch.version.cuda) >= 11.8:
        extra_compile_args["cxx"].append("-DENABLE_FP8")
        extra_compile_args["nvcc"].append("-DENABLE_FP8")


def distribute_as_lib_extension():
    from Cython.Build import cythonize

    ext_modules = []
    for file in get_distribute_as_lib_files():
        module_name = file.replace("/", ".").replace(".py", "")
        ext_modules.append(
            Extension(
                module_name,
                [file],
                extra_compile_args=["-O3"],
            )
        )
    return cythonize(ext_modules, build_dir="build/cython", nthreads=8)


def get_package_version():
    from crossing_cuda_kernels.version import __version__ as public_version
    return str(public_version)


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_wheel_tag():
    torch_version_raw = parse(torch.__version__)
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    torch_cuda_version = parse(torch.version.cuda)
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    wheel_tag = f"cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}"
    return wheel_tag


def get_wheel_url():
    wheel_tag = get_wheel_tag()
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    cuda_kernel_version = get_package_version()

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{cuda_kernel_version}-{wheel_tag}-{python_version}-{python_version}-{platform_name}.whl"

    hash_filename = os.path.join(os.path.dirname(__file__), 'hash.txt')
    if not os.path.exists(hash_filename):
        os.system("git rev-parse HEAD > hash.txt")
    git_sha = open(hash_filename, 'r').read().strip()
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{cuda_kernel_version}", git_sha=git_sha, wheel_name=urllib.parse.quote(wheel_filename))

    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{super().wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()

    @property
    def wheel_dist_name(self) -> str:
        return f"{super().wheel_dist_name}-{get_wheel_tag()}"


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )  # free memory in GB
            max_num_jobs_memory = int(
                free_memory_gb / 9
            )  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

        if not self.parallel:
            self.parallel = min(16, max(1, os.cpu_count() // 2))


ext_modules = []
if not CUDA_HOME and not ROCM_HOME:
    pass
else:
    if _is_cuda():
        ext_modules = [
            # weightonly_gemm_extension(),
        ]

cmdclass = {"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension}
cached_cmdclass = {"bdist_wheel": CachedWheelsCommand}

setup(
    name=PACKAGE_NAME,
    setup_requires=["psutil"],
    use_scm_version={
        "write_to": "crossing_cuda_kernels/version.py",
    },
    packages=find_packages(exclude=("build", "csrc")),
    install_requires=["torch",],
    description="",
    ext_modules=ext_modules,
    cmdclass=cmdclass if ext_modules
    else cached_cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
