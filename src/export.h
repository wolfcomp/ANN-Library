#pragma once
#ifndef DllExport
#define DllExport __declspec(dllexport)
#endif
#ifndef ExternCDllExport
#define ExternCDllExport extern "C" __declspec(dllexport)
#endif