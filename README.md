You may have to install Python with sqlite extensions enabled:

PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions --enable-optimizations --with-openssl=/usr/local/opt/openssl@1.1" LDFLAGS="-L/usr/local/opt/sqlite/lib" CPPFLAGS="-I/usr/local/opt/sqlite/include" pyenv install 3.8.2