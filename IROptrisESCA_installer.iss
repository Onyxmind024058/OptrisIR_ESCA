[Setup]
AppName=Optris IR ESCA
AppVersion=1.0.0
DefaultDirName={autopf}\Optris_IR_ESCA
DefaultGroupName=Optris_IR_ESCA
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=Optris_IR_ESCA_setup
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

; Default is “admin”, but user can choose “current user only”
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog


[Files]
Source: "dist\OptrisIR_ESCA\OptrisIR_ESCA.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\OptrisIR_ESCA\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

; Put SDK in auto appdata (per-user or all-users depending on choice)
Source: "dist\OptrisIR_ESCA\_internal\sdk\*"; DestDir: "{autoappdata}\Optris_IR_ESCA\sdk"; Flags: recursesubdirs createallsubdirs
[Icons]
Name: "{group}\OptrisIR_ESCA"; Filename: "{app}\OptrisIR_ESCA.exe"
Name: "{commondesktop}\OptrisIR_ESCA"; Filename: "{app}\OptrisIR_ESCA.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\OptrisIR_ESCA.exe"; Description: "Launch OptrisIR_ESCA"; Flags: nowait postinstall skipifsilent
