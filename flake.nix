{
  description = "A flake for python development environment, using nix for python package management.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python3;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python and core packages
            (python.withPackages (ps: with ps; [
              numpy
              scipy
              pandas
              scikit-learn
              torch
              fasttext
              pip
              virtualenv
              setuptools
            ]))

            # System and development tools
            graphviz
            antlr
            jdk
            fd
            curl
            unzip
            zsh
          ];

          shellHook = ''
            echo "🚀 Development Environment Initialized"

            # Virtual Environment Setup
            VENV_DIR="./venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python virtual environment in $VENV_DIR"
              python -m venv $VENV_DIR
            fi

            source $VENV_DIR/bin/activate

            # Install additional Python dependencies via pip
            pip install py-solc-x

            # Environment Variables
            export SOLCX_BINARY_PATH=./.solcx/
            export SHELL=$(which zsh)

            # Optional: Display Python version
            python --version
          '';
        };
      }
    );
}
