{
  description = "Live face detection GUI using PyQt and OpenCV";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [];
          config = {
            allowUnfree = true;
          };
        };

        pyPkgs = pkgs.python311Packages;

        mediapipePkg = pyPkgs.buildPythonPackage rec {
          pname = "mediapipe";
          version = "0.10.5";
          format = "wheel";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/86/72/ddcfd72568305c981b11c588209c8e21fd69abbe66dead796bf6eaa03e63/${pname}-${version}-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
            hash = "sha256-toHj0AkmrW5oq/S81pe3j+LNQVfPWUtsmbIixnbCk1c=";
          };
          nativeBuildInputs = [
            pyPkgs.pythonRelaxDepsHook
            pkgs.autoPatchelfHook
          ];
          buildInputs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.glib
            pkgs.libGL
            pkgs.libglvnd
            pkgs.alsa-lib
            pkgs.xorg.libX11
            pkgs.xorg.libXext
            pkgs.xorg.libXfixes
            pkgs.xorg.libXrender
            pkgs.xorg.libxcb
          ];
          pythonRelaxDeps = [
            "numpy"
            "protobuf"
            "opencv-contrib-python"
          ];
          propagatedBuildInputs = with pyPkgs; [
            absl-py
            attrs
            flatbuffers
            numpy
            protobuf
            sounddevice
            matplotlib
            opencv4
          ];
          doCheck = false;
          pythonImportsCheck = [ "mediapipe" ];
        };
        pymupdfPkg = pyPkgs.buildPythonPackage rec {
          pname = "pymupdf";
          version = "1.26.5";
          format = "wheel";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/5b/5a/1292a0df4ff71fbc00dfa8c08759d17c97e1e8ea9277eb5bc5f079ca188d/${pname}-${version}-cp39-abi3-manylinux_2_28_x86_64.whl";
            hash = "sha256-yq0P/rY9zEopykDzxo17eNMqky6DSwBWtSnMC9uq/8k=";
          };
          nativeBuildInputs = [
            pkgs.autoPatchelfHook
          ];
          buildInputs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.freetype
            pkgs.harfbuzz
            pkgs.libjpeg
            pkgs.openjpeg
            pkgs.mesa
            pkgs.xorg.libX11
          ];
          propagatedBuildInputs = with pyPkgs; [
            numpy
          ];
          doCheck = false;
          pythonImportsCheck = [ "fitz" "pymupdf" ];
        };

        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.numpy
          ps.opencv4
          ps.pyqt5
          mediapipePkg
          pymupdfPkg
        ]);

        faceApp = pkgs.writeShellApplication {
          name = "face-gui";
          runtimeInputs = [ pythonEnv pkgs.ffmpeg pkgs.qt5.qtbase.bin ];
          text = ''
            export OPENCV_HAAR_DATA_DIR=${pkgs.opencv4}/share/opencv4/haarcascades
            
            if [ -n "''${PYTHONPATH-}" ]; then
              export PYTHONPATH=${self}/src:''${PYTHONPATH}
            else
              export PYTHONPATH=${self}/src
            fi
            export QT_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins
            export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms
            
            # Install pymupdf from source if needed
            if ! ${pythonEnv}/bin/python -c "import fitz; fitz.mupdf" 2>/dev/null; then
              echo "PyMuPDF build support is limited in nixpkgs - using available version"
            fi
            
            exec ${pythonEnv}/bin/python -m face_gui.app "$@"
          '';
        };
      in {
        packages.default = faceApp;

        apps.default = {
          type = "app";
          program = "${faceApp}/bin/face-gui";
        };

        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv pkgs.ffmpeg pkgs.qt5.full pkgs.qt5.qtbase.bin pkgs.python311.pkgs.pip ];
          shellHook = ''
            export OPENCV_HAAR_DATA_DIR=${pkgs.opencv4}/share/opencv4/haarcascades
            export QT_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins
            export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms
            export PYTHONUNBUFFERED=1
            
            # Install pymupdf from source with mupdf support
            if ! python -c "import fitz; fitz.mupdf" 2>/dev/null; then
              echo "Installing PyMuPDF from source..."
              python -m pip install --user -U --no-binary :all: pymupdf 2>&1 | tail -5
            fi
          '';
        };
      }
    );
}
