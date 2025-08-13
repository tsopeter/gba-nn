SRC_DIR := source
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

CXX := g++
CXXFLAGS := -Wall -O2 -std=c++17

TARGET := app
MACRO  := -DSYSTEM

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(MACRO) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(MACRO) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)