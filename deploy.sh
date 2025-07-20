cd _builder
hugo --verbose --buildDrafts --buildFuture
cd ..
cp -rf _builder/public/* .
rm -rf _builder/public
