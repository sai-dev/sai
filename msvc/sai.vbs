username = InputBox("Inserisci nome utente:","Credenziali per SAI")
password = InputBox("Inserisci password:","Credenziali per SAI")
Set objShell = CreateObject("WScript.Shell")
cmdline = "autogtp --url http://sai.unich.it/ --username "+username+" --password "+password
objShell.Run(cmdline)
