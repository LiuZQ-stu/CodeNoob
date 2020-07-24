import win32gui
import win32con
import win32clipboard as w 
msg = "你个傻逼"

name = "徐思玉"
w.OpenClipboard()
w.EmptyClipboard()
w.SetClipboardData(win32con.CF_UNICODETEXT, msg)
w.CloseClipboard()
handle = win32gui.FindWindow(None, name)
for i in range(10):  
	win32gui.SendMessage(handle, 770, 0, 0)  
	win32gui.SendMessage(handle, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)