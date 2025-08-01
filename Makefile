gocca:
	go fmt ./...
	go install -ldflags "-X main.GitVersion=$(git describe --tags --always --dirty)" ./...

test:
	#. ~/.bashrc && go test -parallel=1 -cover ./...
	go test -p=1 -parallel=1 -cover ./...

bench:
	go test -bench=. -run=^$ -cpu=1 -parallel=1 -benchtime=1x ./...

tidy:
	go mod tidy

generate:
	go generate ./...

occa-install:
	if [ ! -d occa ]; then git clone https://github.com/libocca/occa.git; fi
	cd occa && mkdir -p build && cd build && cmake .. -DOCCA_ENABLE_CUDA=ON -DCMAKE_INSTALL_RPATH=/usr/local/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE -DOCCA_ENABLE_METAL=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && make -j8 && sudo make install
	# For NVIDIA GPU support (requires CUDA toolkit):
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_CUDA=ON
	# For AMD GPU support (requires ROCm):
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_HIP=ON
	# For OpenCL support:
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_OPENCL=ON
	# For OpenMP support:
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_OPENMP=ON
	# Or enable multiple backends:
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    #     -DOCCA_ENABLE_CUDA=ON \
    #     -DOCCA_ENABLE_OPENMP=ON \
    #     -DOCCA_ENABLE_OPENCL=ON
	# Update library cache
	if [ `which ldconfig` ]; then sudo ldconfig; fi
	# Required: Set OCCA installation directory
	export OCCA_DIR=/usr/local
	# Optional: Set cache directory (defaults to ~/.occa if not set)
	export OCCA_CACHE_DIR=$(HOME)/.occa
	make verify-occa-install

verify-occa-install:
	# Add to your shell configuration for permanent setup
	if [ -f $(HOME)/.bashrc ]; then echo 'export OCCA_DIR=/usr/local' >> $(HOME)/.bashrc && echo 'export DYLD_LIBRARY_PATH=/usr/local/lib:$$DYLD_LIBRARY_PATH' >> ~/.bashrc; fi
	# Verify installation and check available backends
	if [ -f $(HOME)/.bashrc ]; then . ~/.bashrc && occa info; fi
