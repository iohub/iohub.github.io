cd _builder
hugo
cd ..
cp -rf _builder/public/* .
rm -rf _builder/public
