!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

!ifndef SOURCE_DIR
  !error "SOURCE_DIR must identify the PyInstaller ProbeFlow directory"
!endif
!ifndef OUTPUT_FILE
  !error "OUTPUT_FILE must identify the installer to create"
!endif
!ifndef ICON_FILE
  !error "ICON_FILE must identify ProbeFlow.ico"
!endif
!ifndef LICENSE_FILE
  !error "LICENSE_FILE must identify the ProbeFlow MIT license"
!endif
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "1.0.0 RC 1"
!endif
!ifndef FILE_VERSION
  !define FILE_VERSION "1.0.0.1"
!endif

Unicode True
ManifestDPIAware True
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 64

Name "ProbeFlow ${PRODUCT_VERSION}"
OutFile "${OUTPUT_FILE}"
InstallDir "$LOCALAPPDATA\Programs\ProbeFlow"
InstallDirRegKey HKCU "Software\SPMQT-Lab\ProbeFlow" "InstallDir"
Icon "${ICON_FILE}"
UninstallIcon "${ICON_FILE}"
BrandingText "ProbeFlow — SPMQT-Lab"

VIProductVersion "${FILE_VERSION}"
VIAddVersionKey /LANG=1033 "CompanyName" "SPMQT-Lab"
VIAddVersionKey /LANG=1033 "FileDescription" "ProbeFlow installer"
VIAddVersionKey /LANG=1033 "FileVersion" "${FILE_VERSION}"
VIAddVersionKey /LANG=1033 "LegalCopyright" "Copyright (C) 2026 SPMQT-Lab and contributors"
VIAddVersionKey /LANG=1033 "ProductName" "ProbeFlow"
VIAddVersionKey /LANG=1033 "ProductVersion" "${PRODUCT_VERSION}"

!define MUI_ABORTWARNING
!define MUI_ICON "${ICON_FILE}"
!define MUI_UNICON "${ICON_FILE}"
!define MUI_FINISHPAGE_RUN "$INSTDIR\ProbeFlow.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch ProbeFlow"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${LICENSE_FILE}"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "ProbeFlow requires 64-bit Windows."
    Abort
  ${EndIf}
FunctionEnd

Section "ProbeFlow" SecMain
  SetShellVarContext current
  SetRegView 64
  SetOutPath "$INSTDIR"
  File /r "${SOURCE_DIR}\*"

  WriteUninstaller "$INSTDIR\Uninstall ProbeFlow.exe"
  CreateShortcut "$SMPROGRAMS\ProbeFlow.lnk" "$INSTDIR\ProbeFlow.exe"

  WriteRegStr HKCU "Software\SPMQT-Lab\ProbeFlow" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "DisplayName" "ProbeFlow"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "Publisher" "SPMQT-Lab"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "DisplayIcon" "$INSTDIR\ProbeFlow.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "UninstallString" "$\"$INSTDIR\Uninstall ProbeFlow.exe$\""
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow" \
    "NoRepair" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  SetRegView 64
  Delete "$SMPROGRAMS\ProbeFlow.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\ProbeFlow"
  DeleteRegKey HKCU "Software\SPMQT-Lab\ProbeFlow"
SectionEnd
