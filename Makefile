CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
LDLIBS   ?= -lzstd

itch_parser: itch_parser.cpp
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

# golden test: C++ output must match the Python reference byte for byte
.PHONY: test
test: itch_parser
	python3 make_test_capture.py
	python3 itch_parser.py testdata --tickers ALL --include-admin --outdir out_py
	./itch_parser    testdata --tickers ALL --include-admin --outdir out_cpp
	cmp out_py/itch_orders_all_all.csv out_cpp/itch_orders_all_all.csv && echo "golden test passed"

.PHONY: clean
clean:
	rm -rf itch_parser out_py out_cpp testdata
