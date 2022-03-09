clean:
	rm -rf build

compile: clean
	mkdir -p build/
	cd build/ && cmake -DVERBOSE=$(VERBOSE) ..
	$(MAKE) -C build/
	$(MAKE) -C build/ install
