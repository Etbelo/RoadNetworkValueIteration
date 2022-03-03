clean:
	$(MAKE) -C lib/ clean
	rm -rf doc
	rm -rf build* debug* release*
	rm -f CMakeLists.txt.user
	rm -rf .vs/

compile: clean
	mkdir -p build/
	cd build/ && cmake -DCMAKE_BUILD_TYPE=Release ..
	$(MAKE) -C build/
	$(MAKE) -C build/ install
