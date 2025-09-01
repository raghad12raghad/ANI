def build_company_summary_vba_module() -> str:
    return r'''Option Explicit

'=========================
' Company Summary: Simple 2-column facts table (Arabic-friendly)
' يجلب بيانات أساسية من Yahoo Finance ويعرضها في جدول بسيط مع تنسيقات أساسية
'=========================
Sub RunCompanySummary()
    On Error GoTo EH
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    Dim tkr As String
    tkr = InputBox("اكتب الرمز (مثال: AAPL أو 1120.SR):", "Company Summary")
    If Len(tkr) = 0 Then GoTo Done

    Dim url As String, js As String
    url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/" & tkr & _
          "?modules=price,assetProfile,summaryDetail,defaultKeyStatistics"
    js = HttpGet(url)
    If Len(js) = 0 Then
        MsgBox "تعذر جلب البيانات. تحقّق من الاتصال أو الرمز.", vbExclamation, "Company Summary"
        GoTo Done
    End If

    '==== استخراج الحقول الأساسية
    Dim name$, sector$, industry$, country$, currency$
    Dim priceVar As Variant, mcapVar As Variant, sharesVar As Variant
    Dim peVar As Variant, betaVar As Variant, wkHiVar As Variant, wkLoVar As Variant

    name = NzStr(JsonFindString(js, "longName"))
    If name = "" Then name = NzStr(JsonFindString(js, "shortName"))
    sector = NzStr(JsonFindString(js, "sector"))
    industry = NzStr(JsonFindString(js, "industry"))
    country = NzStr(JsonFindString(js, "country"))
    currency = NzStr(JsonFindString(js, "currency"))

    priceVar = NzNum(JsonFindRaw(js, "regularMarketPrice"))
    mcapVar  = NzNum(JsonFindRaw(js, "marketCap"))
    sharesVar = NzNum(JsonFindRaw(js, "sharesOutstanding"))
    peVar = NzNum(JsonFindRaw(js, "trailingPE"))
    If IsEmpty(peVar) Then peVar = NzNum(JsonFindRaw(js, "forwardPE"))
    betaVar = NzNum(JsonFindRaw(js, "beta"))
    wkHiVar = NzNum(JsonFindRaw(js, "fiftyTwoWeekHigh"))
    wkLoVar = NzNum(JsonFindRaw(js, "fiftyTwoWeekLow"))

    '==== إنشاء الورقة + تعبئة البيانات
    Dim ws As Worksheet
    Set ws = CreateOrClearSheet("Company Summary")

    With ws
        .Cells.Clear
        .DisplayRightToLeft = True         ' اتجاه يمين-إلى-يسار لملاءمة العربية
        .Cells.Font.Name = "Segoe UI"
        .Cells.Font.Size = 10

        ' العناوين
        .Range("A1").Value = "الحقل / Field"
        .Range("B1").Value = "القيمة / Value"
        With .Range("A1:B1")
            .Font.Bold = True
            .Interior.Color = RGB(14, 165, 233) ' sky-500
            .Font.Color = vbWhite
            .RowHeight = 22
            .HorizontalAlignment = xlCenter
            .VerticalAlignment = xlCenter
        End With

        Dim r As Long: r = 2
        r = PutRow(ws, r, "Ticker", tkr)
        r = PutRow(ws, r, "Company Name", name)
        r = PutRow(ws, r, "Sector / Industry", sector & " / " & industry)
        r = PutRow(ws, r, "Country", country)
        r = PutRow(ws, r, "Currency", currency)
        r = PutRow(ws, r, "Price", priceVar)
        r = PutRow(ws, r, "Market Cap", mcapVar)
        r = PutRow(ws, r, "Shares Outstanding", sharesVar)
        r = PutRow(ws, r, "P/E (trailing/forward)", peVar)
        r = PutRow(ws, r, "Beta", betaVar)
        r = PutRow(ws, r, "52W High", wkHiVar)
        r = PutRow(ws, r, "52W Low", wkLoVar)

        ' تكوين جدول Excel من النطاق
        On Error Resume Next
        .ListObjects("SummaryTable").Unlist
        If Err.Number <> 0 Then Err.Clear
        On Error GoTo 0

        Dim lastRow As Long: lastRow = r - 1
        Dim lo As ListObject
        Set lo = .ListObjects.Add(SourceType:=xlSrcRange, _
                                  Source:=.Range("A1:B" & lastRow), _
                                  XlListObjectHasHeaders:=xlYes)
        lo.Name = "SummaryTable"

        On Error Resume Next
        lo.TableStyle = "TableStyleMedium2"
        If Err.Number <> 0 Then Err.Clear: lo.TableStyle = "TableStyleMedium9"
        On Error GoTo 0

        ' تنسيق الأرقام تلقائياً في العمود B (عملة/فواصل)
        FormatIfNumeric .Range("B2:B" & lastRow), currency

        .Columns("A:B").AutoFit
        .Range("A2").Select
        ActiveWindow.FreezePanes = True

        ' إعدادات الطباعة
        With .PageSetup
            .Orientation = xlPortrait
            .Zoom = False
            .FitToPagesWide = 1
            .FitToPagesTall = False
            .LeftMargin = Application.InchesToPoints(0.4)
            .RightMargin = Application.InchesToPoints(0.4)
            .TopMargin = Application.InchesToPoints(0.5)
            .BottomMargin = Application.InchesToPoints(0.5)
            .CenterHorizontally = True
        End With
    End With

Done:
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    If Len(js) > 0 Then MsgBox "تم إنشاء ملخص الشركة في ورقة 'Company Summary'.", vbInformation, "Done"
    Exit Sub
EH:
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    MsgBox "Error: " & Err.Number & " - " & Err.Description, vbCritical, "RunCompanySummary"
End Sub

'=========================
' HTTP (late-binding)
'=========================
Private Function HttpGet(ByVal url As String) As String
    On Error GoTo EH
    Dim x As Object
    Set x = CreateObject("MSXML2.XMLHTTP")
    x.Open "GET", url, False
    x.setRequestHeader "User-Agent", "Mozilla/5.0"
    x.send
    If x.Status = 200 Then HttpGet = CStr(x.responseText) Else HttpGet = ""
    Exit Function
EH:
    HttpGet = ""
End Function

'=========================
' JSON helpers (lightweight)
' ملاحظة: نقرأ من حقول Yahoo "raw" أو السلاسل مباشرة بدون مراجع JSON خارجية
'=========================
Private Function JsonFindRaw(ByVal js As String, ByVal key As String) As String
    Dim pat As String: pat = """" & key & """:{" & """raw"":"
    Dim p As Long: p = InStr(1, js, pat, vbTextCompare)
    If p = 0 Then Exit Function
    p = p + Len(pat)
    Dim i As Long: i = p
    Dim ch As String, buf As String
    Do While i <= Len(js)
        ch = Mid$(js, i, 1)
        If (ch Like "[0-9.-]") Or ch = "E" Or ch = "e" Or ch = "+" Then
            buf = buf & ch
        Else
            Exit Do
        End If
        i = i + 1
    Loop
    JsonFindRaw = Trim$(buf)
End Function

Private Function JsonFindString(ByVal js As String, ByVal key As String) As String
    Dim pat As String: pat = """" & key & """:"""
    Dim p As Long: p = InStr(1, js, pat, vbTextCompare)
    If p = 0 Then Exit Function
    p = p + Len(pat)
    Dim i As Long: i = p
    Dim ch As String, buf As String
    Do While i <= Len(js)
        ch = Mid$(js, i, 1)
        If ch = """" Then Exit Do
        If ch = "\" Then
            i = i + 1
            If i <= Len(js) Then ch = Mid$(js, i, 1)
        End If
        buf = buf & ch
        i = i + 1
    Loop
    JsonFindString = buf
End Function

Private Function NzStr(ByVal s As String) As String
    If Len(Trim$(s)) = 0 Then NzStr = "" Else NzStr = Trim$(s)
End Function

' يحوّل النص لرقم إن أمكن، وإلا يعيد Empty
Private Function NzNum(ByVal s As String) As Variant
    On Error GoTo EH
    If Len(Trim$(s)) = 0 Then
        NzNum = Empty
    ElseIf IsNumeric(s) Then
        NzNum = CDbl(s)
    Else
        NzNum = Empty
    End If
    Exit Function
EH:
    NzNum = Empty
End Function

'=========================
' تنسيق وكتابة الصفوف
'=========================
Private Function PutRow(ws As Worksheet, ByVal r As Long, ByVal label As String, ByVal v As Variant) As Long
    ws.Cells(r, 1).Value = label
    If IsEmpty(v) Then
        ws.Cells(r, 2).Value = ""
    Else
        ws.Cells(r, 2).Value = v
    End If
    PutRow = r + 1
End Function

Private Sub FormatIfNumeric(rng As Range, Optional ByVal curr As String = "")
    On Error Resume Next
    Dim c As Range
    For Each c In rng.Cells
        If IsNumeric(c.Value) Then
            If InStr(1, LCase$(curr), "usd") > 0 Or curr = "USD" Then
                c.NumberFormat = "$#,##0.00"
            ElseIf curr <> "" Then
                c.NumberFormat = "#,##0.00" ' عملة غير محدّدة الرمز
            Else
                c.NumberFormat = "#,##0.00"
            End If
        End If
    Next c
End Sub

'=========================
' Helper: إنشاء/تفريغ الورقة
'=========================
Private Function CreateOrClearSheet(ByVal sheetName As String) As Worksheet
    On Error Resume Next
    Set CreateOrClearSheet = ThisWorkbook.Worksheets(sheetName)
    If CreateOrClearSheet Is Nothing Then
        Set CreateOrClearSheet = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
        CreateOrClearSheet.Name = sheetName
    Else
        CreateOrClearSheet.Cells.Clear
    End If
    On Error GoTo 0
End Function
'''
