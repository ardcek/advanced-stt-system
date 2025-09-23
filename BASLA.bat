@echo off
chcp 65001 >nul
title Ses Transkripsiyon Uygulamasi - Mehmet Arda Cekic
color 0A
echo.
echo  ===========================================
echo   🎙️ SES TRANSKRIPSIYON UYGULAMASI 🎙️
echo         Made by Mehmet Arda Cekic
echo  ===========================================
echo.
echo  Lütfen bir seçenek yazın:
echo.
echo  [1] Mevcut ses dosyası analiz et
echo  [2] Canlı kayıt yap (ENTER ile dur)  
echo  [3] Belirli süre kayıt yap
echo  [4] Test kaydı (30 saniye)
echo  [5] Çıkış
echo.
set /p choice="Seçiminiz (1-5): "

cd /d "C:\Users\Arda\Desktop\test"

if "%choice%"=="1" goto file_analysis
if "%choice%"=="2" goto live_record
if "%choice%"=="3" goto timed_record
if "%choice%"=="4" goto test_record  
if "%choice%"=="5" goto end

:file_analysis
echo.
set /p filename="Ses dosyasi adi (ornek: meeting.wav): "
set /p title="Toplanti basligi: "
echo.
echo 🔄 Analiz basliyor... (Lutfen bekleyin)
C:\Users\Arda\Desktop\test\venv\Scripts\python.exe main.py --file "%filename%" --stt large-v3 --title "%title%"
echo.
echo ✅ Analiz tamamlandi! Dosyalar olusturuldu:
echo    📄 transcript.txt - Tam metin
echo    📋 summary.txt - Ozet
echo    ✅ tasks.txt - Gorevler  
echo    📄 notes.md - Yapilandirilmis notlar
echo    📊 meeting_minutes.docx - Word belgesi
pause
goto menu

:live_record
echo.
set /p title="Kayit basligi: "
echo.
echo 🔴 CANLI KAYIT BASLIYOR...
echo ⚠️  ENTER'a basarak durdurun!
C:\Users\Arda\Desktop\test\venv\Scripts\python.exe main.py --stream --stt large-v3 --title "%title%"
echo.
echo ✅ Kayit ve analiz tamamlandi!
pause
goto menu

:timed_record
echo.
set /p duration="Kayit suresi (saniye): "
set /p title="Kayit basligi: "
echo.
echo 🔴 %duration% SANİYE KAYIT BASLIYOR...
C:\Users\Arda\Desktop\test\venv\Scripts\python.exe main.py --duration %duration% --stt medium --title "%title%"
echo.
echo ✅ Kayit ve analiz tamamlandi!
pause
goto menu

:test_record
echo.
echo 🧪 30 SANİYE TEST KAYDI BASLIYOR...
C:\Users\Arda\Desktop\test\venv\Scripts\python.exe main.py --duration 30 --stt small --title "Test Kaydi"
echo.
echo ✅ Test tamamlandi!
pause
goto menu

:menu
cls
echo.
echo  ===========================================
echo   🎙️  SES TRANSKRIPSYON UYGULAMASI  🎙️
echo  ===========================================
echo.
echo  Lutfen bir secenek yazin:
echo.
echo  [1] Mevcut ses dosyasi analiz et
echo  [2] Canli kayit yap (ENTER ile dur)
echo  [3] Belirli sure kayit yap
echo  [4] Test kaydi (30 saniye)
echo  [5] Cikis
echo.
set /p choice="Seciminiz (1-5): "

if "%choice%"=="1" goto file_analysis
if "%choice%"=="2" goto live_record
if "%choice%"=="3" goto timed_record
if "%choice%"=="4" goto test_record
if "%choice%"=="5" goto end
goto menu

:end
echo.
echo 👋 Tesekkurler! Uygulamadan cikiliyor...
timeout /t 2 >nul
exit