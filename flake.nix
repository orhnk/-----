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
          config.allowUnfree = true;
        };

        basePyPkgs = pkgs.python311Packages;

        pyPkgs = basePyPkgs.overrideScope (final: prev: {
          mediapipe = final.buildPythonPackage rec {
            pname = "mediapipe";
            version = "0.10.5";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/86/72/ddcfd72568305c981b11c588209c8e21fd69abbe66dead796bf6eaa03e63/${pname}-${version}-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
              hash = "sha256-toHj0AkmrW5oq/S81pe3j+LNQVfPWUtsmbIixnbCk1c=";
            };
            nativeBuildInputs = [
              final.pythonRelaxDepsHook
              pkgs.autoPatchelfHook
            ];
            buildInputs = with pkgs; [
              stdenv.cc.cc.lib
              zlib
              glib
              libGL
              libglvnd
              alsa-lib
              xorg.libX11
              xorg.libXext
              xorg.libXfixes
              xorg.libXrender
              xorg.libxcb
            ];
            pythonRelaxDeps = [
              "numpy"
              "protobuf"
              "opencv-contrib-python"
            ];
            propagatedBuildInputs = with final; [
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

          pymupdf = final.buildPythonPackage rec {
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
            buildInputs = with pkgs; [
              stdenv.cc.cc.lib
              zlib
              freetype
              harfbuzz
              libjpeg
              openjpeg
              mesa
              xorg.libX11
            ];
            propagatedBuildInputs = [ final.numpy ];
            doCheck = false;
            pythonImportsCheck = [ "fitz" "pymupdf" ];
          };
        });

        pythonEnv = pyPkgs.python.withPackages (ps:
          with ps; [
            pip
            numpy
            opencv4
            pyqt5
            mediapipe
            pymupdf
          ]
        );

        musettirApp = pyPkgs.buildPythonApplication rec {
          pname = "musettir";
          version = "0.1.0";
          format = "pyproject";
          src = ./.;
          nativeBuildInputs = [
            pyPkgs.setuptools
            pyPkgs.wheel
          ];
          propagatedBuildInputs = with pyPkgs; [
            numpy
            opencv4
            pyqt5
            mediapipe
            pymupdf
          ];
          pythonImportsCheck = [ "musettir.app" ];
          doCheck = false;
        };

        # Wrap the musettir binary with Qt environment variables
        faceGuiApp = pkgs.writeShellApplication {
          name = "musettir";
          runtimeInputs = [ musettirApp pkgs.qt5.qtbase.bin pkgs.qt5.qtwayland.bin pkgs.opencv4 ];
          text = ''
            export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms:${pkgs.qt5.qtwayland.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms"
            export QT_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins"
            export OPENCV_HAAR_DATA_DIR="${pkgs.opencv4}/share/opencv4/haarcascades"
            exec ${musettirApp}/bin/musettir "$@"
          '';
        };

        # Create a package with desktop entry for system integration
        musettirPackage = pkgs.stdenv.mkDerivation {
          pname = "musettir";
          version = "0.1.0";
          
          dontUnpack = true;
          dontBuild = true;

          installPhase = ''
            mkdir -p $out/bin
            mkdir -p $out/share/applications
            mkdir -p $out/share/pixmaps
            
            cp ${faceGuiApp}/bin/musettir $out/bin/musettir
            chmod +x $out/bin/musettir
            
            # Install desktop entry
            cat > $out/share/applications/musettir.desktop <<'EOF'
[Desktop Entry]
Type=Application
Name=Musettir
Comment=Live face detection GUI using PyQt5 and OpenCV
Exec=$out/bin/musettir
Icon=camera-photo
Categories=Graphics;Video;
Terminal=false
EOF
          '';

          meta = with pkgs.lib; {
            description = "Live face detection GUI using PyQt5 and OpenCV";
            license = licenses.gpl3;
            platforms = platforms.linux;
          };
        };
      in {
        packages = {
          default = musettirPackage;
          musettir = musettirPackage;
        };

        apps.default = {
          type = "app";
          program = "${faceGuiApp}/bin/musettir";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.ffmpeg
            pkgs.qt5.qtwayland
          ];
          shellHook = ''
            export OPENCV_HAAR_DATA_DIR=${pkgs.opencv4}/share/opencv4/haarcascades
            export QT_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins
            export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.qtbase}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms:${pkgs.qt5.qtwayland}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms
            export QT_QPA_PLATFORM=''${QT_QPA_PLATFORM:-xcb}
            export PYTHONUNBUFFERED=1
          '';
        };
      }
    );
}
