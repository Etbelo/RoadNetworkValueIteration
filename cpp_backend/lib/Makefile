clean: 
	rm -rf build/ install/
	rm -rf .vs/

compile: clean
	mkdir -p build/
	cd build/ && cmake ..
	$(MAKE) -C build/
	$(MAKE) -C build/ install
