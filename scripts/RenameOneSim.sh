# Example command line parameters: E 001 00129
cd /share/testde/ucapwhi/GowerStreetSimsFlat
tar -xvf ../GowerStreetSims/runs$1/run$2.tar.gz -C .
mv -v ./run$2/ ./sim$3/
tar -czvf sim$3.tar.gz ./sim$3/
rm -rfv ./sim$3/

