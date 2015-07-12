#!/bin/bash


cd csa
CC=/usr/bin/gcc ./configure && make && make install
cd ..

cd nn
CC=/usr/bin/gcc ./configure && make && make install
cd ..

cd gridutils
CC=/usr/bin/gcc ./configure && make && make install
cd ..

cd gridgen
CC=/usr/bin/gcc ./configure && make && make lib && make shlib && make install
cd ..
