[Setup]
AppName=Optris IR ESCA
AppVersion=1.0.0
AppPublisher=Universität Basel/ Paul Hiret
AppPublisherURL=mailto:paul.hiret@unibas.ch
DefaultDirName={pf}\Optris_IR_ESCA
DefaultGroupName=Optris_IR_ESCA
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=Optris_IR_ESCA_setup
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "dist\OptrisIR_ESCA\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OptrisIR_ESCA"; Filename: "{app}\OptrisIR_ESCA.exe"
Name: "{commondesktop}\OptrisIR_ESCA"; Filename: "{app}\OptrisIR_ESCA.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\OptrisIR_ESCA.exe"; Description: "Launch OptrisIR_ESCA"; Flags: nowait postinstall skipifsilent
