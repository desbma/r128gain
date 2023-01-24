WINEARCH          ?= win32
FFMPEG_VERSION    ?= 4.2
FFMPEG_CACHE_DIR  ?= ${HOME}/.cache/ci/r128gain/ffmpeg-${WINEARCH}

# versions
PYTHON_VERSION       := 3.8.10
PYTHON_VERSION_MAJOR := $(word 1,$(subst ., ,${PYTHON_VERSION})).$(word 2,$(subst ., ,${PYTHON_VERSION}))
PYTHON_VERSION_SHORT := $(subst .,,${PYTHON_VERSION_MAJOR})
CXFREEZE_VERSION     := 6.14.0
SOX_VERSION          := 14.4.2

# installers & packages
DOWNLOAD_DIR            ?= dl
PYTHON_EMBEDDABLE_ARCHIVE-win32 := ${DOWNLOAD_DIR}/python-${PYTHON_VERSION}-embed-win32.zip
PYTHON_EMBEDDABLE_ARCHIVE-win64 := ${DOWNLOAD_DIR}/python-${PYTHON_VERSION}-embed-amd64.zip
PYTHON_EMBEDDABLE_ARCHIVE       := ${PYTHON_EMBEDDABLE_ARCHIVE-${WINEARCH}}
GET_PIP                         := ${DOWNLOAD_DIR}/get-pip.py
FUTURE_WHEEL                    := ${DOWNLOAD_DIR}/future-0.17.1-py3-none-any.whl
CXFREEZE_WHEEL-win32            := ${DOWNLOAD_DIR}/cx_Freeze-${CXFREEZE_VERSION}-cp${PYTHON_VERSION_SHORT}-cp${PYTHON_VERSION_SHORT}-${WINEARCH}.whl
CXFREEZE_WHEEL-win64            := ${DOWNLOAD_DIR}/cx_Freeze-${CXFREEZE_VERSION}-cp${PYTHON_VERSION_SHORT}-cp${PYTHON_VERSION_SHORT}-win_amd64.whl
CXFREEZE_WHEEL                  := ${CXFREEZE_WHEEL-${WINEARCH}}
7ZIP_INSTALLER                  := ${DOWNLOAD_DIR}/7z2107.msi
SOX_ZIP                         := ${DOWNLOAD_DIR}/sox-${SOX_VERSION}-win32.zip

# native env & tools
CURL     := curl -s -f -L -m 600 --connect-timeout 30 --retry 10 --retry-max-time 5
WINE_DIR := /tmp/$(subst /,_,${CURDIR})/wine-${WINEARCH}
UNZIP    := unzip -q -o -DD

# installed files
PYTHON_INSTALLED   := ${WINE_DIR}/drive_c/Python${PYTHON_VERSION_SHORT}/python.exe
PIP_INSTALLED      := $(dir ${PYTHON_INSTALLED})Scripts/pip.exe
FUTURE_INSTALLED   := $(dir ${PYTHON_INSTALLED})Lib/site-packages/future/__init__.py
CXFREEZE_INSTALLED := $(dir ${PYTHON_INSTALLED})Scripts/cxfreeze
7ZIP_INSTALLED       := ${WINE_DIR}/drive_c/7-Zip/7z.sfx
SOX_INSTALLED        := ${WINE_DIR}/drive_c/windows/system32/sox.exe
FFMPEG_INSTALLED     := ${WINE_DIR}/drive_c/windows/system32/ffmpeg.exe
ifeq ($(shell lsb_release -si),Ubuntu)
WINPTHREAD_DLL_win32 := /usr/i686-w64-mingw32/lib/libwinpthread-1.dll
WINPTHREAD_DLL_win64 := /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll
else
WINPTHREAD_DLL_win32 := /usr/i686-w64-mingw32/bin/libwinpthread-1.dll
WINPTHREAD_DLL_win64 := /usr/x86_64-w64-mingw32/bin/libwinpthread-1.dll
endif
WINPTHREAD_DLL       := ${WINPTHREAD_DLL_${WINEARCH}}

# wine env
WINE        := WINEPREFIX=${WINE_DIR} WINEARCH=${WINEARCH} WINEDLLOVERRIDES="mscoree,mshtml=" WINEDEBUG=-all wine
WINE_PYTHON := ${WINE} ${PYTHON_INSTALLED}
WINE_PIP    := ${WINE} ${PIP_INSTALLED}

# temp dirs
OUTPUT_DIR   := r128gain-${WINEARCH}


.PHONY: all test test_ut test_freeze fclean clean


all:: r128gain_${WINEARCH}.7z r128gain_${WINEARCH}.zip r128gain_${WINEARCH}.exe

test:: test_ut test_freeze

print-%:
	@echo $*=$($*)


#
# End targets
#

r128gain_${WINEARCH}.7z: ${OUTPUT_DIR}/r128gain.exe ${OUTPUT_DIR}/LICENSE ${OUTPUT_DIR}/readme.txt ${OUTPUT_DIR}/ffmpeg.exe ${OUTPUT_DIR}/libwinpthread-1.dll
	7z a -t7z -mx9 $@ ${OUTPUT_DIR} > /dev/null

r128gain_${WINEARCH}.zip: ${OUTPUT_DIR}/r128gain.exe ${OUTPUT_DIR}/LICENSE ${OUTPUT_DIR}/readme.txt ${OUTPUT_DIR}/ffmpeg.exe
	7z a -tzip -mx9 $@ ${OUTPUT_DIR} > /dev/null

r128gain_${WINEARCH}.exe: ${7ZIP_INSTALLED} r128gain_${WINEARCH}.7z
	cat $^ > $@


#
# Intermediate targets
#

${OUTPUT_DIR}/r128gain.exe: ${CXFREEZE_INSTALLED} ${LXML_INSTALLED} .requirements
	mkdir -p $(dir $@)
	cd .. && ${WINE} ${PYTHON_INSTALLED} freeze.py build_exe -b win/${OUTPUT_DIR} 2> /dev/null > /dev/null
	touch $@

${OUTPUT_DIR}/LICENSE:
	mkdir -p $(dir $@)
	cp ../LICENSE $@

${OUTPUT_DIR}/readme.txt:
	mkdir -p $(dir $@)
	echo 'Get source, help and latest version at: https://github.com/desbma/r128gain' > $@

${OUTPUT_DIR}/ffmpeg.exe: ${FFMPEG_INSTALLED}
	mkdir -p $(dir $@)
	cp $< $@

${OUTPUT_DIR}/libwinpthread-1.dll: ${WINPTHREAD_DLL}
	mkdir -p $(dir $@)
	cp $< $@


#
# Tests
#

test_ut: ${LXML_INSTALLED} .requirements ${FFMPEG_INSTALLED} ${SOX_INSTALLED}
	cd .. && ${WINE} ${PYTHON_INSTALLED} setup.py test

test_freeze: ${OUTPUT_DIR}/r128gain.exe ${OUTPUT_DIR}/python${PYTHON_VERSION_SHORT}.dll ${OUTPUT_DIR}/ffmpeg.exe ${OUTPUT_DIR}/libwinpthread-1.dll sample.ogg sample.opus
	${WINE} $< -v debug sample.ogg sample.opus

sample.ogg: ${DOWNLOAD_DIR}/sample.ogg
	cp $< $@

sample.opus: ${DOWNLOAD_DIR}/sample.opus
	cp $< $@


#
# Installations
#

.requirements: ${PIP_INSTALLED} ${FUTURE_INSTALLED}
	cd .. && ${WINE_PIP} -qq install -r requirements.txt
	cd .. && ${WINE_PIP} -qq install .
	touch $@

${PYTHON_INSTALLED}: ${PYTHON_EMBEDDABLE_ARCHIVE}
	mkdir -p $(dir $@)
	${UNZIP} -d $(dir $@) $<
	${UNZIP} -d $(dir $@) $(dir $@)python${PYTHON_VERSION_SHORT}.zip
	rm $(dir $@)python${PYTHON_VERSION_SHORT}.zip
	echo 'import site' >>$(dir $@)python${PYTHON_VERSION_SHORT}._pth

${PIP_INSTALLED}: ${PYTHON_INSTALLED} ${GET_PIP}
	${WINE_PYTHON} ${GET_PIP} -qq
	touch $@

${FUTURE_INSTALLED}: ${FUTURE_WHEEL} ${PIP_INSTALLED}
	${WINE_PIP} -qq install $<
	touch $@

${CXFREEZE_INSTALLED}: ${CXFREEZE_WHEEL} ${PIP_INSTALLED}
	${WINE_PIP} -qq install $<
	touch $@

${7ZIP_INSTALLED}: ${7ZIP_INSTALLER}
	${WINE} msiexec /i $< /qb INSTALLDIR='c:\7-Zip'
	touch $@

${FFMPEG_INSTALLED}:
	mkdir -p $(dir $@)
	./cross-build-minimal-ffmpeg ${FFMPEG_VERSION} ${WINEARCH} ${FFMPEG_CACHE_DIR} $@

${SOX_INSTALLED}: ${SOX_ZIP}
	mkdir -p ${WINE_DIR}/drive_c
	${UNZIP} $< -d ${WINE_DIR}/drive_c
	cp ${WINE_DIR}/drive_c/sox-${SOX_VERSION}/*.dll $(dir $@)
	cp ${WINE_DIR}/drive_c/sox-${SOX_VERSION}/sox.exe $@


#
# Downloads
#

${PYTHON_EMBEDDABLE_ARCHIVE}:
	mkdir -p $(dir $@)
	${CURL} https://www.python.org/ftp/python/${PYTHON_VERSION}/$(notdir $@) > $@

${GET_PIP}:
	mkdir -p $(dir $@)
	${CURL} https://bootstrap.pypa.io/$(notdir $@) > $@

${FUTURE_WHEEL}:
	mkdir -p $(dir $@)
	${CURL} https://www.piwheels.org/simple/future/$(notdir $@) > $@

${CXFREEZE_WHEEL-win32}:
	mkdir -p $(dir $@)
	${CURL} https://files.pythonhosted.org/packages/00/6f/f05b5d50e1932c33d8aa8b0af255c5fc3f3830e6af1a82529509be0a1490/$(notdir $@) > $@

${CXFREEZE_WHEEL-win64}:
	mkdir -p $(dir $@)
	${CURL} https://files.pythonhosted.org/packages/0d/bd/6ccd8b1f5ba4a12fbb5cc43dc53fa3741fb7b776b07c58d4bbb94fabc590/$(notdir $@) > $@

${7ZIP_INSTALLER}:
	mkdir -p $(dir $@)
	${CURL} http://www.7-zip.org/a/$(notdir $@) > $@.unckecked
	echo "103210153e60b4234015796bb5f12483f99b5909df8c2fe5c9d3a823d4bdc721602a5261ad794e5280ff9f0d5f79add4e2a732dfb087fe8b4844d789acb8ea42  $@.unckecked" | sha512sum --status -c - && mv $@.unckecked $@

${SOX_ZIP}:
	mkdir -p $(dir $@)
	${CURL} https://sourceforge.net/projects/sox/files/sox/${SOX_VERSION}/$(notdir $@) > $@

${DOWNLOAD_DIR}/sample.ogg:
	mkdir -p $(dir $@)
	${CURL} https://upload.wikimedia.org/wikipedia/en/0/09/Opeth_-_Deliverance.ogg > $@

${DOWNLOAD_DIR}/sample.opus:
	mkdir -p $(dir $@)
	${CURL} https://opus-codec.org/static/examples/ehren-paper_lights-96.opus > $@


#
# Clean
#

fclean:
	-rm -Rf ${WINE_DIR}
	-rm -Rf ${OUTPUT_DIR} .requirements
	-rm -f r128gain_${WINEARCH}.7z r128gain_${WINEARCH}.zip r128gain_${WINEARCH}.exe sample.ogg

clean: fclean
	-rm -Rf ${DOWNLOAD_DIR}
