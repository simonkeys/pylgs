SHELL = /bin/zsh

pylgsversion = 0.1.0
pythonversion = 3.10
envname = conda-$(pythonversion)-pylgs

env : 
	conda create -n $(envname) -c conda-forge --yes python=$(pythonversion)
	conda run -n $(envname) python -m pip install -e ".[dev]"
	conda run -n $(envname) python -m ipykernel install --user --name $(envname) # https://stackoverflow.com/a/72395091/2576097
	nbdev_install_hooks

remove-env:
	conda remove -n $(envname) --all --yes
	jupyter kernelspec uninstall $(envname) -y

zip :
	zip -r ../pylgs-$(pylgsversion).zip . -x '.*' '**/.*'

build :
	python -m build

test :
	clear
	nbdev_test --path nbs/api --timing 

testall :
	clear
	nbdev_test --timing 

pre :
	nbdev_export
	nbdev_clean
	nbdev_readme --chk_time

deploy-pypi:
	conda run -n $(envname) python -m pip install twine
	nbdev_pypi


# sundials :
# 	mkdir build-sundials-6.5.1
# 	cd ..
# 	cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_SIZE=64 -DCMAKE_INSTALL_PREFIX=/usr/local/ ../sundials-6.5.1/
# 	sudo make install