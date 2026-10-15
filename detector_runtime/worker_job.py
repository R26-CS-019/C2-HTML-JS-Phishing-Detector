"""Windows job ownership: closing the job stops the worker and its children."""
import ctypes
from ctypes import wintypes


class WindowsJob:
    def __init__(self,process):
        class Basic(ctypes.Structure):
            _fields_=[('ProcessUserTime',ctypes.c_longlong),('JobUserTime',ctypes.c_longlong),
                      ('LimitFlags',wintypes.DWORD),('MinWorkingSet',ctypes.c_size_t),
                      ('MaxWorkingSet',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                      ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),
                      ('SchedulingClass',wintypes.DWORD)]
        class Counters(ctypes.Structure):
            _fields_=[(n,ctypes.c_ulonglong) for n in ['ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes']]
        class Extended(ctypes.Structure):
            _fields_=[('Basic',Basic),('IO',Counters),('ProcessMemory',ctypes.c_size_t),
                      ('JobMemory',ctypes.c_size_t),('PeakProcessMemory',ctypes.c_size_t),
                      ('PeakJobMemory',ctypes.c_size_t)]
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype=wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.api.SetInformationJobObject.restype=wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype=wintypes.BOOL
        self.api.CloseHandle.argtypes=[wintypes.HANDLE]
        self.api.CloseHandle.restype=wintypes.BOOL
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info=Extended();info.Basic.LimitFlags=0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info))
                or not self.api.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(process._handle)))):
            error=ctypes.get_last_error();self.close()
            raise ctypes.WinError(error)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle);self.handle=None
