NVCC := nvcc
EXT := .bin

.PHONY: run build clean

# usage: make run FILE=matmul.cu
# expands to nvcc matmul.cu -o matmul.bin && ./matmul.bin
run:
	@test -n "$(FILE)" || { echo "usage: make run FILE=<source>.cu"; exit 2; }
	@stripped_name=$(basename -- "$(FILE)" .cu); out=$$stripped_name$(EXT); \
		$(NVCC) "$(FILE)" -o "$$out" && "./$$out"

# usage: make build FILE=matmul.cu
build:
	@test -n "$(FILE)" || { echo "usage: make build FILE=<source>.cu"; exit 2; }
	@stripped_name=$(basename -- "$(FILE)" .cu); out=$$stripped_name$(EXT); \
		$(NVCC) "$(FILE)" -o "$$out"

clean:
	@rm -f *$(EXT)


# SHORT TUTORIAL
#
# to create a make variable
# name := value
#	to access a make variable
#	$(name)
#
# ---------------------------
#
# to create a shell variable
# @name=value
# to access a shell variable
# $name <- from the shell
# $$name <- from the makefile
# 
# example shell command that uses variable
# file=matmul.cu; nvcc $file 
# hence when the shell sees $name it looks
# for a variable with that name.
#
#
# $$ in a make file means one $ should be
# passed to the shell
# hence if you want the shell to see $file
# in the makefile write $$file
#
# @ is used to hide a command on the shell
# so echo "hello" in a make file will produce
# ```
#	echo "hello"
#	hello
# ```
# but 
# @echo "hello" will produce
# ```
# hello
# ```
