{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  outputs = inputs @ { self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
  let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = false;
      config.allowUnsupportedSystem = true;
    };
    python = pkgs.python312;
    systemPackages = with pkgs; [
      taglib
      openssl
      git
      libxml2
      libxslt
      libzip
      zlib
      stdenv.cc.cc.lib
      stdenv.cc
      ncurses5
      binutils
      gitRepo gnupg autoconf curl
      procps gnumake util-linux m4 gperf unzip
      libGLU libGL
      glib
      freeglut
      gcc
    ];
    ultralytics-python = python.pkgs.buildPythonPackage {
      pname = "ultralytics";
      version = "8.3.49";
      format = "pyproject";
          #doCheck = false;
      propagatedBuildInputs = [
        python.pkgs.setuptools
        python.pkgs.seaborn
        python.pkgs.pyyaml
        python.pkgs.requests
        python.pkgs.scipy
        python.pkgs.torch
        python.pkgs.torchvision
        python.pkgs.tqdm
        python.pkgs.psutil
        python.pkgs.py-cpuinfo
        python.pkgs.opencv4
        python.pkgs.matplotlib
        python.pkgs.pandas
        clip-python
        hub-sdk-python
        ultralytics-thop-python
        #opencv-python
      ];
      src = pkgs.fetchFromGitHub {
        owner = "ultralytics";
        repo = "ultralytics";
        rev = "v8.3.49";
        sha256 = "sha256-SRfLxuYrLBdqwZbDMYmqUvEh92qdZseFllFBhUjB59M=";
      };
      patchPhase = ''
        substituteInPlace pyproject.toml --replace "\"opencv-python>=4.6.0\"," ""
        substituteInPlace pyproject.toml --replace "\"torchvision>=0.9.0\"," ""
      '';
    };
    tensorflow-python = python.pkgs.buildPythonPackage rec {
      pname = "tensorflow";
      version = "2.12.0";
      format = "wheel";
      #doCheck = false;
      propagatedBuildInputs = [
        python.pkgs.setuptools
      ];
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/6b/6d/00ef55900312ba8c8aa9d64b13c74fb8cba0afa24cc4a2ce74796f530244/tensorflow-2.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
        sha256 = "sha256-bsSik06hnpLyepZo7OQwJe1e/hS10ZvlOwdpK8ikGJ0=";
      };
    };
    model-compression-toolkit-python = python.pkgs.buildPythonPackage rec {
      pname = "model_compression_toolkit";
      version = "2.1.1";
      format = "setuptools";
      #doCheck = false;
      preBuild = "touch requirements.txt";
      propagatedBuildInputs = [
        python.pkgs.torch
        python.pkgs.setuptools
        python.pkgs.poetry-core
      ];
      src = python.pkgs.fetchPypi {
        pname = pname;
        version = version;
        sha256 = "sha256-JXlnSqo5yWV+pmzvU2EqKNIMOMHxTklIvO8I2K9HJyc=";
      };
          #srs = pkgs.fetchurl {
          #  url = "https://files.pythonhosted.org/packages/8b/56/69b3c68ca4cd5c9d3881f78c8a9978e27c75ff5c36c5811f3a38c5c87e82/model_compression_toolkit-2.1.1-py3-none-any.whl";
          #  sha256 = "sha256-Ea10xTl9u2UaIySI2dpQIGar08Oc8DLBJtiXRv1QpAE=";
          #};
    };
    sony-custom-layers-python = python.pkgs.buildPythonPackage rec {
      pname = "sony_custom_layers";
      version = "0.2.0";
      format = "wheel";
      #doCheck = false;
      propagatedBuildInputs = [
        python.pkgs.requests
        python.pkgs.setuptools
        python.pkgs.poetry-core
        python.pkgs.wheel
      ];
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/43/03/021a00c0470b147581ca3eb30a6fec2fa74d08ce676538e9c8a55a2082b6/sony_custom_layers-0.2.0-py3-none-any.whl";
        sha256 = "sha256-FXDgpFtoDnrETpMa5Q3No9wONSjTvz2TyTacihQS4wM=";
      };
    };
    sam2-python = python.pkgs.buildPythonPackage {
      pname = "sam2";
      version = "1.0";
      #format = "wheel";
          #doCheck = false;
      propagatedBuildInputs = [
        python.pkgs.torch
        python.pkgs.torchvision
        python.pkgs.hydra-core
        python.pkgs.iopath
      ];
      src = pkgs.fetchFromGitHub {
        owner = "facebookresearch";
        repo = "sam2";
        rev = "c2ec8e14a185632b0a5d8b161928ceb50197eddc";
        hash = "sha256-eSekGCY+5VG3XOYvFN+4THhKpN7AMW0br8PuAAtCRP4=";
      };
    };
    azure-ai-vision-imageanalysis-python = python.pkgs.buildPythonPackage {
      pname = "azure-ai-vision-imageanalysis";
      version = "1.0.0b2";
      format = "pyproject";
      #doCheck = false;
      propagatedBuildInputs = [
        python.pkgs.setuptools
        python.pkgs.poetry-core
        python.pkgs.azure-core
        python.pkgs.isodate
      ];
      src = python.pkgs.fetchPypi {
        pname = "azure-ai-vision-imageanalysis";
        version = "1.0.0b2";
        sha256 = "sha256-cEy5nj1AUToB0GI8JTekZ/m3/DsHaiWXunFJkrHx5qs=";
      };
    };
    flash-attn-python = python.pkgs.buildPythonPackage rec {
      pname = "flash-attention";
      version = "2.3.6";
      pyproject = true;
      src = pkgs.fetchFromGitHub {
        owner = "Dao-AILab";
        repo = "flash-attention";
        rev = "v${version}";
        hash = "sha256-7wCB8EGL9Ioqq38+UHINQuwyFbnJkk7CXfujoKPdPl8=";
        fetchSubmodules = true;
      };
          #doCheck = false;
      dontUseCmakeConfigure = true;
      CUDA_HOME = pkgs.symlinkJoin {
        name = "cuda-redist";
        paths = buildInputs;
      };
      build-tools = [
        python.pkgs.setuptools
        python.pkgs.wheel
      ];
          #propagatedBuildInputs = [
          #  python.pkgs.requests
          #  python.pkgs.setuptools
          #  python.pkgs.torch-bin
          #  python.pkgs.packaging
          #  python.pkgs.psutil
          #  python.pkgs.einops
          #  pkgs.cudaPackages.cudatoolkit
          #  pkgs.cudaPackages.cudnn
          #  pkgs.ninja
          #];
      nativeBuildInputs = [
        pkgs.which
        pkgs.git
            #pkgs.gcc11
            #pkgs.cmake
        pkgs.ninja
            #pkgs.cudaPackages.cuda_nvcc
      ];
      buildInputs = with pkgs.cudaPackages; [
        cuda_cudart # cuda_runtime_api.h
        libcusparse # cusparse.h
        cuda_cccl # nv/target
        libcublas # cublas_v2.h
        libcusolver # cusolverDn.h
        libcurand # curand_kernel.h
        cuda_nvcc
      ];
      dependencies = [
        python.pkgs.torch
        python.pkgs.psutil
        python.pkgs.ninja
        python.pkgs.einops
      ];
      pythonImportsCheck = [ "flash_attn" ];
      MAX_JOBS = "2";
    };
    hub-sdk-python = python.pkgs.buildPythonPackage {
      pname = "hub-sdk";
      version = "0.0.7";
      format = "pyproject";
      propagatedBuildInputs = [ python.pkgs.requests python.pkgs.setuptools ];
      src = python.pkgs.fetchPypi {
        pname = "hub-sdk";
        version = "0.0.7";
        sha256 = "sha256-njUVSHSCS/QoDbdrXKmfCuscmEbkBPVnxefcc3ofaWw=";
      };
    };
    ultralytics-thop-python = python.pkgs.buildPythonPackage rec {
      pname = "ultralytics_thop";
      version = "2.0.12";
      format = "pyproject";
      propagatedBuildInputs = [
        python.pkgs.requests
        python.pkgs.setuptools
        python.pkgs.numpy
        python.pkgs.torch
      ];
      doCheck = false;
      src = python.pkgs.fetchPypi {
        pname = pname;
        version = version;
        sha256 = "sha256-M1hgwQGRxbM5cOsfXrfkcsGXCV86mc+1RoezlRcdZC0=";
      };
    };
    opencv-python = pkgs.python3.pkgs.buildPythonPackage {
      pname = "opencv";
      version = "4.9.0.80";
      format = "pyproject";
      propagatedBuildInputs = [ pkgs.python311Packages.requests pkgs.python311Packages.setuptools pkgs.python311Packages.scikit-build ];
      src = pkgs.python.pkgs.fetchPypi {
        pname = "opencv-python";
        version = "4.9.0.80";
        sha256 = "sha256-Gp8OYmfeOhodsMVCE9Aix8i1ucpLWA6AvcWFFskiyeE=";
      };
    };
    supervision-python = pkgs.python3.pkgs.buildPythonPackage {
      pname = "supervision";
      version = "0.19.0";
      format = "pyproject";
      propagatedBuildInputs = [ pkgs.python311Packages.setuptools pkgs.python311Packages.poetry-core ];
      src = pkgs.python.pkgs.fetchPypi {
        pname = "supervision";
        version = "0.19.0";
        sha256 = "sha256-b7zIfsLoU4jwCwYDtQt4pIAD45kk4XijMjtNo/MlWc4=";
      };
    };
    roboflow-python = pkgs.python3.pkgs.buildPythonPackage {
      pname = "roboflow";
      version = "1.1.27";
      format = "pyproject";
      propagatedBuildInputs = [ pkgs.python311Packages.setuptools pkgs.python311Packages.poetry-core ];
      doCheck = false;
      src = pkgs.fetchFromGitHub {
        repo = "roboflow-python";
        owner = "roboflow";
        rev = "v1.1.27";
        sha256 = "sha256-9U9BiOb4OQcCkueN9LzlxQxYhH9ufcJQYU4g1lBwqAU=";
      };
    };
    clip-python = python.pkgs.buildPythonPackage rec {
      pname = "clip-anytorch";
      version = "2.5.2";
      format = "setuptools";
      doCheck = false;
      propagatedBuildInputs = [
        python.pkgs.ftfy
        python.pkgs.regex
      ];
      src = pkgs.fetchFromGitHub {
        owner = "rom1504";
        repo = "CLIP";
        rev = version;
        hash = "sha256-EqVkpMQHawoCFHNupf49NrvLdGCq35wnYBpdP81Ztd4=";
      };
    };
    transformers = python.pkgs.buildPythonPackage rec {
      pname = "transformers";
      version = "4.46.3";
          #format = "wheel";
      doCheck = false;
      src = python.pkgs.fetchPypi {
        pname = "transformers";
        version = "4.46.3";
        sha256 = "sha256-juSzrpQ/4z6Cr/+Og39LBSBYsHypvjy1tyntMSlfcsw=";
      };
      propagatedBuildInputs = [
        python.pkgs.filelock
        python.pkgs.huggingface-hub
        python.pkgs.numpy
        python.pkgs.packaging
        python.pkgs.pyyaml
        python.pkgs.regex
        python.pkgs.requests
        python.pkgs.tokenizers
        python.pkgs.safetensors
        python.pkgs.tqdm
        python.pkgs.torch
      ];
    };
    onnxslim-python = python.pkgs.buildPythonPackage rec {
      pname = "onnxslim";
      version = "0.1.44";
      format = "pyproject";
      propagatedBuildInputs = [
        python.pkgs.pip
        python.pkgs.setuptools
        python.pkgs.poetry-core
        python.pkgs.onnx
        python.pkgs.sympy
      ];
      src = python.pkgs.fetchPypi {
        pname = pname;
        version = version;
        sha256 = "sha256-tAq0NbjXsA99O60dglNrQP2hi1zKEj4OcBPclsNnXl0=";
      };
          #src = pkgs.fetchFromGitHub {
          #  owner = "inisis";
          #  repo = "OnnxSlim";
          #  rev = version;
          #  hash = "sha256-v8JSuYYBwEtN36zUEr6atGFzAr8QmJhXYfImuyGBBvg=";
          #};
    };
    huggingface-hub-python = python.pkgs.buildPythonPackage rec {
      pname = "huggingface-hub-python";
      version = "0.25.2";
      pyproject = true;
      src = pkgs.fetchFromGitHub {
        owner = "huggingface";
        repo = "huggingface_hub";
        tag = "v${version}";
        hash = "sha256-wUZkh8ti35F8bGtEKJZGSRySLvCkIWEFajQNWo/WCys=";
      };
      build-system = [ python.pkgs.setuptools ];
      dependencies = [
        python.pkgs.filelock
        python.pkgs.fsspec
        python.pkgs.packaging
        python.pkgs.pyyaml
        python.pkgs.requests
        python.pkgs.tqdm
        python.pkgs.typing-extensions
      ];
      doCheck = false;
      pythonImportsCheck = [ "huggingface_hub" ];
    };
    super-image-python = python.pkgs.buildPythonPackage {
      pname = "super-image";
      version = "0.1.7";
      format = "pyproject";
      propagatedBuildInputs = [
        python.pkgs.poetry-core
        python.pkgs.requests
        python.pkgs.setuptools
        python.pkgs.torch
        python.pkgs.torchvision
        python.pkgs.opencv4
        python.pkgs.h5py
        python.pkgs.tqdm
        huggingface-hub-python
      ];
      doCheck = false;
      doInstallCheck = false;
      patchPhase = ''
        substituteInPlace pyproject.toml --replace "opencv-python = \">= 4.5.2.54\"" ""
        substituteInPlace pyproject.toml --replace "torchvision = \">= 0.10.0\"" ""
      '';
      src = python.pkgs.fetchPypi {
        pname = "super_image";
        version = "0.1.7";
        sha256 = "sha256-St1Abcn0lbs+83BHyIwg5Tybb/eFyKsSk8LkcqX/Zi0=";
      };
    };
  in {
    devShells.default = pkgs.mkShell {
      name = "yolo";
      buildInputs = systemPackages ++ [
        ( pkgs.writeScriptBin "fittings" "./inference.py -svpd" )
        python
        python.pkgs.setuptools
        python.pkgs.requests
            #google-cloud-sdk
            #python.pkgs.google-cloud-vision
            #python.pkgs.transformers
            #flash-attn-python
            #python.pkgs.sentencepiece
            #python.pkgs.accelerate
            #python.pkgs.requests-toolbelt
            #python.pkgs.openai
            #python.pkgs.shapely
            #python.pkgs.pyclipper
            #python.pkgs.scikit-image
            #python.pkgs.xlsxwriter
        python.pkgs.python-dotenv
        python.pkgs.flask
        python.pkgs.onnx
        python.pkgs.onnxruntime
        python.pkgs.pylibdmtx
        python.pkgs.distutils
        python.pkgs.easyocr
        python.pkgs.symspellpy
        python.pkgs.pytesseract
        python.pkgs.pyzbar
        super-image-python
            #onnxslim-python
        python.pkgs.torch
            #python.pkgs.torchvision
            #python.pkgs.tensorflow-bin
        (python.pkgs.opencv4.override { enableContrib = true; })
            #python.pkgs.timm
            #python.pkgs.einops
            #(python311Packages.onnxruntime.override { cudaSupport = true; })
            #ultralytics-python
            #sam2-python
        (python.pkgs.rapidocr-onnxruntime.overridePythonAttrs { pythonImportsCheck = []; doCheck = false; })
            #model-compression-toolkit-python
            #sony-custom-layers-python
            #tensorflow-python
      ];
    };
  });
}
