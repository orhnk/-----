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

        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.numpy
          ps.opencv4
          ps.pyqt5
          mediapipePkg
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
          packages = [ pythonEnv pkgs.ffmpeg pkgs.qt5.full pkgs.qt5.qtbase.bin ];
          shellHook = ''
            export OPENCV_HAAR_DATA_DIR=${pkgs.opencv4}/share/opencv4/haarcascades
            export QT_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins
            export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms
            export PYTHONUNBUFFERED=1
          '';
        };
      }
    );
}
