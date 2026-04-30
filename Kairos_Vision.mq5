//+------------------------------------------------------------------+
//|              Kairos_Vision.mq5   v6.00                           |
//+------------------------------------------------------------------+
#property copyright   "Kairos"
#property version     "6.00"
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

#property indicator_label1  "EMA Fast"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrCyan
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "EMA Slow"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrMagenta
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

input int    InpEMAFast   = 8;
input int    InpEMASlow   = 21;
input int    InpRSIPeriod = 7;
input int    InpMACDFast  = 5;
input int    InpMACDSlow  = 13;
input int    InpMACDSig   = 3;
input int    InpATRPeriod = 7;
input int    InpOBLook    = 60;
input int    InpSwingW    = 4;
input int    InpSMCLook   = 200;
input double InpMinScore  = 60.0;
input double InpVolSpike  = 1.5;

// ── Цветова схема ──────────────────────────────────────────────────
// EMA Fast = CYAN, EMA Slow = MAGENTA
// OB Bullish = GOLD рамка, OB Bearish = VIOLET рамка
// BOS↑ = LIME стрелка, BOS↓ = RED стрелка
// CHOCH↑ = DEEP SKY BLUE ромб, CHOCH↓ = ORANGE RED ромб
// EQH = TOMATO пунктир, EQL = AQUA пунктир

double BufFast[], BufSlow[];
int    hRSI, hMACD, hATR;
const  string PFX = "PTV5_";

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufFast, INDICATOR_DATA);
   SetIndexBuffer(1, BufSlow, INDICATOR_DATA);
   ArraySetAsSeries(BufFast, false);
   ArraySetAsSeries(BufSlow, false);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, 0.0);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, 0.0);

   hRSI  = iRSI (_Symbol, PERIOD_CURRENT, InpRSIPeriod, PRICE_CLOSE);
   hMACD = iMACD(_Symbol, PERIOD_CURRENT, InpMACDFast, InpMACDSlow, InpMACDSig, PRICE_CLOSE);
   hATR  = iATR (_Symbol, PERIOD_CURRENT, InpATRPeriod);

   // Ярки свещи — контрастни на OB цветовете
   ChartSetInteger(0, CHART_COLOR_BACKGROUND,  C'5,5,15');
   ChartSetInteger(0, CHART_COLOR_CANDLE_BULL,  clrLimeGreen);
   ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR,  clrRed);
   ChartSetInteger(0, CHART_COLOR_CHART_UP,     clrLimeGreen);
   ChartSetInteger(0, CHART_COLOR_CHART_DOWN,   clrRed);
   ChartSetInteger(0, CHART_COLOR_GRID,         C'18,18,30');
   ChartSetInteger(0, CHART_COLOR_FOREGROUND,   clrSilver);

   IndicatorSetString(INDICATOR_SHORTNAME, "⚡ Kairos Vision");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, PFX);
   IndicatorRelease(hRSI);
   IndicatorRelease(hMACD);
   IndicatorRelease(hATR);
}

//+------------------------------------------------------------------+
// Swing finder: non-series (0=oldest, last=newest)
//+------------------------------------------------------------------+
int FindSwings(const double &arr[], int n, int w, int lookBars,
               bool hi, double &P[], int &I[], int mx)
{
   int cnt=0;
   int from=MathMax(w, n-lookBars);
   int to  =n-w-1;
   for(int i=from; i<=to && cnt<mx; i++)
   {
      bool ok=true;
      for(int k=-w; k<=w && ok; k++){
         if(k==0) continue;
         int kk=i+k; if(kk<0||kk>=n) continue;
         if(hi  && arr[i]<arr[kk]) ok=false;
         if(!hi && arr[i]>arr[kk]) ok=false;
      }
      if(ok){ P[cnt]=arr[i]; I[cnt]=i; cnt++; }
   }
   return cnt;
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],  const double &high[],
                const double &low[],   const double &close[],
                const long &tick_volume[], const long &volume[],
                const int &spread[])
{
   if(rates_total < 30) return 0;

   //── EMA (ръчно, non-series) ───────────────────────────────────────
   double af=2.0/(InpEMAFast+1), as2=2.0/(InpEMASlow+1);
   int st=prev_calculated>1?prev_calculated-1:1;
   if(prev_calculated==0){ BufFast[0]=close[0]; BufSlow[0]=close[0]; }
   for(int i=st;i<rates_total;i++){
      BufFast[i]=af*close[i]+(1.0-af)*BufFast[i-1];
      BufSlow[i]=as2*close[i]+(1.0-as2)*BufSlow[i-1];
   }

   int    last=rates_total-1;
   double curC=close[last], curO=open[last];
   double emaF=BufFast[last], emaS=BufSlow[last];

   //── Индикатори ────────────────────────────────────────────────────
   double rsi=50; double rb[1];
   if(CopyBuffer(hRSI,0,0,1,rb)==1) rsi=rb[0];

   double macdH=0; double mm[1],ms[1];
   if(CopyBuffer(hMACD,0,0,1,mm)==1&&CopyBuffer(hMACD,1,0,1,ms)==1) macdH=mm[0]-ms[0];

   double atr=curC*0.001; double ab[1];
   if(CopyBuffer(hATR,0,0,1,ab)==1&&ab[0]>0) atr=ab[0];

   double vsum=0; int vp=MathMin(20,rates_total);
   for(int i=0;i<vp;i++) vsum+=(double)tick_volume[last-i];
   double vAvg=vsum/vp;
   double vRatio=(vAvg>0)?(double)tick_volume[last]/vAvg:1.0;

   //── Score ─────────────────────────────────────────────────────────
   int bs=0,ss=0;
   if(rsi<40) bs+=20; else if(rsi>60) ss+=20;
   if(macdH>0) bs+=15; else if(macdH<0) ss+=15;
   if(emaF>emaS) bs+=25; else ss+=25;
   if(MathAbs(curC-curO)>atr*0.5){ if(curC>curO) bs+=15; else ss+=15; }
   if(vRatio>InpVolSpike){ if(curC>curO) bs+=15; else ss+=15; }

   //── Изчистване ────────────────────────────────────────────────────
   ObjectsDeleteAll(0, PFX+"OB");
   ObjectsDeleteAll(0, PFX+"BOS");
   ObjectsDeleteAll(0, PFX+"CHO");
   ObjectsDeleteAll(0, PFX+"EQL");
   ObjectsDeleteAll(0, PFX+"SIG");

   //── ORDER BLOCKS — немитигирани, макс 6 ─────────────────────────
   {
      int look   = MathMin(InpOBLook, rates_total-5);
      int obCount= 0;

      for(int i=3; i<look && obCount<6; i++)
      {
         int idx=last-i;
         if(idx<2||idx+1>=rates_total) continue;

         double body=MathAbs(open[idx]-close[idx]);
         if(body<atr*0.3) continue;   // значима свещ

         bool isBear=(open[idx]>close[idx]);   // червена → bullish OB
         bool isBull=(close[idx]>open[idx]);   // зелена  → bearish OB
         double obHigh=high[idx], obLow=low[idx];

         string nm=""; color clr; string lbl="";

         if(isBear){
            // Bullish OB: следващата затваря над тялото на OB свещта
            if(close[idx+1] <= open[idx]) continue;  // над open (не само над high)
            // Немитигиран: close не влиза в зоната
            bool mit=false;
            for(int j=idx+2;j<=last;j++){
               if(close[j]<=obHigh && close[j]>=obLow){mit=true;break;}
            }
            if(mit) continue;
            nm=PFX+"OB_B_"+IntegerToString(idx);
            clr=C'218,165,32'; lbl="OB↑"; bs+=12;
         }
         else if(isBull){
            // Bearish OB: следващата затваря под тялото на OB свещта
            if(close[idx+1] >= open[idx]) continue;
            bool mit=false;
            for(int j=idx+2;j<=last;j++){
               if(close[j]<=obHigh && close[j]>=obLow){mit=true;break;}
            }
            if(mit) continue;
            nm=PFX+"OB_S_"+IntegerToString(idx);
            clr=C'148,0,211'; lbl="OB↓"; ss+=12;
         }
         if(nm=="") continue;

         obCount++;

         // Правоъгълник до текущата свещ
         if(ObjectCreate(0,nm,OBJ_RECTANGLE,0,time[idx],obHigh,time[last],obLow)){
            ObjectSetInteger(0,nm,OBJPROP_COLOR,     clr);
            ObjectSetInteger(0,nm,OBJPROP_WIDTH,     2);
            ObjectSetInteger(0,nm,OBJPROP_FILL,      false);
            ObjectSetInteger(0,nm,OBJPROP_BACK,      false);
            ObjectSetInteger(0,nm,OBJPROP_SELECTABLE,false);
         }
         // Надпис вътре в правоъгълника — горе вляво
         string lnm=nm+"_L";
         double lblY = isBear ? obHigh : obLow;
         if(ObjectCreate(0,lnm,OBJ_TEXT,0,time[idx],lblY)){
            ObjectSetString (0,lnm,OBJPROP_TEXT,     lbl);
            ObjectSetInteger(0,lnm,OBJPROP_COLOR,    clr);
            ObjectSetInteger(0,lnm,OBJPROP_FONTSIZE, 8);
            ObjectSetString (0,lnm,OBJPROP_FONT,     "Arial Bold");
            ObjectSetInteger(0,lnm,OBJPROP_SELECTABLE,false);
         }
      }
   }

   //── BOS / CHOCH ───────────────────────────────────────────────────
   string bosTxt="", chochTxt="";
   {
      double shP[60]; int shI[60];
      double slP[60]; int slI[60];
      int shC=FindSwings(high,rates_total,InpSwingW,InpSMCLook,true, shP,shI,60);
      int slC=FindSwings(low, rates_total,InpSwingW,InpSMCLook,false,slP,slI,60);

      // BOS↑ — пробив нагоре на swing high
      for(int s=0;s<shC;s++){
         double lvl=shP[s]; int si=shI[s];
         for(int i=si+1;i<rates_total;i++){
            if(close[i]>lvl){
               // Пунктирана линия от swing до пробива
               string tl=PFX+"BOS_TH_"+IntegerToString(s);
               if(ObjectCreate(0,tl,OBJ_TREND,0,time[si],lvl,time[i],lvl)){
                  ObjectSetInteger(0,tl,OBJPROP_COLOR,    C'0,180,0');
                  ObjectSetInteger(0,tl,OBJPROP_STYLE,    STYLE_DOT);
                  ObjectSetInteger(0,tl,OBJPROP_WIDTH,    1);
                  ObjectSetInteger(0,tl,OBJPROP_RAY_RIGHT,false);
                  ObjectSetInteger(0,tl,OBJPROP_SELECTABLE,false);
                  ObjectSetInteger(0,tl,OBJPROP_BACK,     true);
               }
               // Стрелка
               string an=PFX+"BOS_AH_"+IntegerToString(s);
               if(ObjectCreate(0,an,OBJ_ARROW_UP,0,time[i],low[i]-atr)){
                  ObjectSetInteger(0,an,OBJPROP_COLOR,    clrLime);
                  ObjectSetInteger(0,an,OBJPROP_WIDTH,    3);
                  ObjectSetInteger(0,an,OBJPROP_SELECTABLE,false);
               }
               // Текст
               string tn=PFX+"BOS_TXT_H"+IntegerToString(s);
               Txt(tn,time[i],lvl,"BOS↑",clrLime,9);  // точно на нивото
               if(s==shC-1){ bs+=20; bosTxt="BOS↑"; }
               break;
            }
         }
      }

      // BOS↓ — пробив надолу на swing low
      for(int s=0;s<slC;s++){
         double lvl=slP[s]; int si=slI[s];
         for(int i=si+1;i<rates_total;i++){
            if(close[i]<lvl){
               string tl=PFX+"BOS_TL_"+IntegerToString(s);
               if(ObjectCreate(0,tl,OBJ_TREND,0,time[si],lvl,time[i],lvl)){
                  ObjectSetInteger(0,tl,OBJPROP_COLOR,    C'180,0,0');
                  ObjectSetInteger(0,tl,OBJPROP_STYLE,    STYLE_DOT);
                  ObjectSetInteger(0,tl,OBJPROP_WIDTH,    1);
                  ObjectSetInteger(0,tl,OBJPROP_RAY_RIGHT,false);
                  ObjectSetInteger(0,tl,OBJPROP_SELECTABLE,false);
                  ObjectSetInteger(0,tl,OBJPROP_BACK,     true);
               }
               string an=PFX+"BOS_AL_"+IntegerToString(s);
               if(ObjectCreate(0,an,OBJ_ARROW_DOWN,0,time[i],high[i]+atr)){
                  ObjectSetInteger(0,an,OBJPROP_COLOR,    clrRed);
                  ObjectSetInteger(0,an,OBJPROP_WIDTH,    3);
                  ObjectSetInteger(0,an,OBJPROP_SELECTABLE,false);
               }
               string tn=PFX+"BOS_TXT_L"+IntegerToString(s);
               Txt(tn,time[i],lvl,"BOS↓",clrRed,9);  // точно на нивото
               if(s==slC-1){ ss+=20; bosTxt="BOS↓"; }
               break;
            }
         }
      }

      //── CHOCH ─────────────────────────────────────────────────────────
      // CHOCH↑ = Lower High (LH) пробит нагоре  → структурата се обръща bullish
      // CHOCH↓ = Higher Low (HL) пробит надолу  → структурата се обръща bearish
      {
         int cc=0; // брояч за ограничение

         // CHOCH↑: намираме всеки LH и търсим пробив нагоре
         for(int s=1; s<shC && cc<8; s++)
         {
            if(shP[s] >= shP[s-1]) continue;   // не е LH → skip
            double lvl=shP[s];
            int    si =shI[s];
            for(int i=si+1; i<rates_total; i++)
            {
               if(close[i]>lvl)
               {
                  string nm=PFX+"CHO_U_"+IntegerToString(s);
                  Txt(nm,time[i],lvl,"◆CHOCH↑",clrDeepSkyBlue,10);  // на нивото
                  // Линия САМО от swing до пробива (не до края на чарта)
                  string hl=PFX+"CHO_HU_"+IntegerToString(s);
                  if(ObjectCreate(0,hl,OBJ_TREND,0,time[si],lvl,time[i],lvl)){
                     ObjectSetInteger(0,hl,OBJPROP_COLOR,     clrDeepSkyBlue);
                     ObjectSetInteger(0,hl,OBJPROP_STYLE,     STYLE_DASH);
                     ObjectSetInteger(0,hl,OBJPROP_WIDTH,     2);
                     ObjectSetInteger(0,hl,OBJPROP_RAY_RIGHT, false);
                     ObjectSetInteger(0,hl,OBJPROP_BACK,      true);
                     ObjectSetInteger(0,hl,OBJPROP_SELECTABLE,false);
                  }
                  cc++;
                  if(s==shC-1){ bs+=25; chochTxt="CHOCH↑"; }
                  break;
               }
            }
         }

         cc=0;
         // CHOCH↓: намираме всеки HL и търсим пробив надолу
         for(int s=1; s<slC && cc<8; s++)
         {
            if(slP[s] <= slP[s-1]) continue;   // не е HL → skip
            double lvl=slP[s];
            int    si =slI[s];
            for(int i=si+1; i<rates_total; i++)
            {
               if(close[i]<lvl)
               {
                  string nm=PFX+"CHO_D_"+IntegerToString(s);
                  Txt(nm,time[i],lvl,"◆CHOCH↓",clrOrangeRed,10);  // на нивото
                  // Линия САМО от swing до пробива
                  string hl=PFX+"CHO_HD_"+IntegerToString(s);
                  if(ObjectCreate(0,hl,OBJ_TREND,0,time[si],lvl,time[i],lvl)){
                     ObjectSetInteger(0,hl,OBJPROP_COLOR,     clrOrangeRed);
                     ObjectSetInteger(0,hl,OBJPROP_STYLE,     STYLE_DASH);
                     ObjectSetInteger(0,hl,OBJPROP_WIDTH,     2);
                     ObjectSetInteger(0,hl,OBJPROP_RAY_RIGHT, false);
                     ObjectSetInteger(0,hl,OBJPROP_BACK,      true);
                     ObjectSetInteger(0,hl,OBJPROP_SELECTABLE,false);
                  }
                  cc++;
                  if(s==slC-1){ ss+=25; chochTxt="CHOCH↓"; }
                  break;
               }
            }
         }
      }

      //── EQUAL HIGHS & LOWS ────────────────────────────────────────
      double tol=atr*0.25;
      int w2=InpSwingW, lk=MathMin(80,rates_total-w2-1);
      int fr=MathMax(w2,rates_total-lk);
      for(int i=fr;i<rates_total-w2;i++){
         bool isH=true,isL=true;
         for(int k=-w2;k<=w2;k++){
            if(k==0) continue;
            int kk=i+k; if(kk<0||kk>=rates_total) continue;
            if(high[i]<high[kk]) isH=false;
            if(low [i]>low [kk]) isL=false;
         }
         if(isH) for(int j=i+w2+2;j<rates_total-w2;j++){
            bool ok=true;
            for(int k=-w2;k<=w2;k++){
               if(k==0) continue;
               int kk=j+k; if(kk<0||kk>=rates_total) continue;
               if(high[j]<high[kk]){ok=false;break;}
            }
            if(ok&&MathAbs(high[i]-high[j])<=tol){
               double lv=(high[i]+high[j])/2.0;
               string nm=PFX+"EQL_H"+IntegerToString(i);
               if(ObjectCreate(0,nm,OBJ_TREND,0,time[i],lv,time[j],lv)){
                  ObjectSetInteger(0,nm,OBJPROP_COLOR,    clrTomato);
                  ObjectSetInteger(0,nm,OBJPROP_STYLE,    STYLE_DASH);
                  ObjectSetInteger(0,nm,OBJPROP_WIDTH,    1);
                  ObjectSetInteger(0,nm,OBJPROP_RAY_RIGHT,false);
                  ObjectSetInteger(0,nm,OBJPROP_SELECTABLE,false);
               }
               Txt(nm+"_L",time[j],lv+atr*0.3,"EQH",clrTomato,8);
               break;
            }
         }
         if(isL) for(int j=i+w2+2;j<rates_total-w2;j++){
            bool ok=true;
            for(int k=-w2;k<=w2;k++){
               if(k==0) continue;
               int kk=j+k; if(kk<0||kk>=rates_total) continue;
               if(low[j]>low[kk]){ok=false;break;}
            }
            if(ok&&MathAbs(low[i]-low[j])<=tol){
               double lv=(low[i]+low[j])/2.0;
               string nm=PFX+"EQL_L"+IntegerToString(i);
               if(ObjectCreate(0,nm,OBJ_TREND,0,time[i],lv,time[j],lv)){
                  ObjectSetInteger(0,nm,OBJPROP_COLOR,    clrAqua);
                  ObjectSetInteger(0,nm,OBJPROP_STYLE,    STYLE_DASH);
                  ObjectSetInteger(0,nm,OBJPROP_WIDTH,    1);
                  ObjectSetInteger(0,nm,OBJPROP_RAY_RIGHT,false);
                  ObjectSetInteger(0,nm,OBJPROP_SELECTABLE,false);
               }
               Txt(nm+"_L",time[j],lv-atr*0.3,"EQL",clrAqua,8);
               break;
            }
         }
      }
   }

   //── Финален сигнал ────────────────────────────────────────────────
   int    fScore=MathMax(bs,ss);
   int    tot=bs+ss;
   double bPct=(tot>0)?(double)bs/tot*100:50;
   double sPct=(tot>0)?(double)ss/tot*100:50;
   string dir="WAIT";
   if(bPct>=60&&fScore>=(int)InpMinScore) dir="BUY";
   else if(sPct>=60&&fScore>=(int)InpMinScore) dir="SELL";

   //── BUY/SELL стрелка ─────────────────────────────────────────────
   ObjectDelete(0,PFX+"SIG_A"); ObjectDelete(0,PFX+"SIG_L");
   if(dir!="WAIT"){
      bool buy=(dir=="BUY");
      double ap=buy?curC-atr*2.0:curC+atr*2.0;
      if(ObjectCreate(0,PFX+"SIG_A",buy?OBJ_ARROW_UP:OBJ_ARROW_DOWN,0,time[last],ap)){
         ObjectSetInteger(0,PFX+"SIG_A",OBJPROP_COLOR,buy?clrLime:clrRed);
         ObjectSetInteger(0,PFX+"SIG_A",OBJPROP_WIDTH,5);
         ObjectSetInteger(0,PFX+"SIG_A",OBJPROP_SELECTABLE,false);
      }
      Txt(PFX+"SIG_L",time[last],buy?curC-atr*3.5:curC+atr*3.5,
          (buy?"▲ BUY ":"▼ SELL ")+IntegerToString(fScore),
          buy?clrLime:clrRed,11);
   }

   //── Dashboard ─────────────────────────────────────────────────────
   color  dc=(dir=="BUY")?clrLime:(dir=="SELL")?clrRed:clrYellow;
   DashPanel(PFX+"BG",8,25,252,222);
   DashLabel(PFX+"T0",17, 36,"⚡ KAIROS VISION",       clrWhite,   9);
   DashLabel(PFX+"T1",17, 52,"────────────────────",  C'50,50,90',8);
   DashLabel(PFX+"T2",17, 64,
      StringFormat("RSI: %.1f  %s",rsi,(rsi<40)?"[OS]":(rsi>60)?"[OB]":"[--]"),
      (rsi<40)?clrLime:(rsi>60)?clrRed:clrSilver,8);
   DashLabel(PFX+"T3",17, 80,
      StringFormat("MACD: %.5f  %s",macdH,(macdH>0)?"[BULL]":"[BEAR]"),
      (macdH>0)?clrLime:clrRed,8);
   DashLabel(PFX+"T4",17, 96,
      StringFormat("EMA %d/%d: %s",InpEMAFast,InpEMASlow,(emaF>emaS)?"[UP]":"[DOWN]"),
      (emaF>emaS)?clrLime:clrRed,8);
   DashLabel(PFX+"T5",17,112,
      StringFormat("Vol: %.1fx  %s",vRatio,(vRatio>InpVolSpike)?"[SPIKE]":"[--]"),
      (vRatio>InpVolSpike)?clrOrange:clrSilver,8);
   DashLabel(PFX+"T6",17,128,"────────────────────",  C'50,50,90',8);
   DashLabel(PFX+"T7",17,140,
      (bosTxt!="")?bosTxt:"BOS:   —",
      (bosTxt!="")?clrLime:clrGray,8);
   DashLabel(PFX+"T8",17,154,
      (chochTxt!="")?chochTxt:"CHOCH: —",
      (chochTxt=="CHOCH↓")?clrOrangeRed:(chochTxt=="CHOCH↑")?clrDeepSkyBlue:clrGray,8);
   DashLabel(PFX+"T9",17,170,"────────────────────",  C'50,50,90',8);
   DashLabel(PFX+"T10",17,182,
      StringFormat("Score: %d  (B:%d / S:%d)",fScore,bs,ss),clrSilver,8);
   DashLabel(PFX+"T11",17,200,"► "+dir,dc,11);

   ChartRedraw(0);
   return rates_total;
}

//+------------------------------------------------------------------+
void DashLabel(string nm,int x,int y,string txt,color c,int fs)
{
   if(ObjectFind(0,nm)<0) ObjectCreate(0,nm,OBJ_LABEL,0,0,0);
   ObjectSetInteger(0,nm,OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0,nm,OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0,nm,OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetString (0,nm,OBJPROP_TEXT,      txt);
   ObjectSetInteger(0,nm,OBJPROP_COLOR,     c);
   ObjectSetInteger(0,nm,OBJPROP_FONTSIZE,  fs);
   ObjectSetString (0,nm,OBJPROP_FONT,      "Consolas");
   ObjectSetInteger(0,nm,OBJPROP_BACK,      false);
   ObjectSetInteger(0,nm,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,nm,OBJPROP_ZORDER,    10);
}

void DashPanel(string nm,int x,int y,int w,int h)
{
   if(ObjectFind(0,nm)<0) ObjectCreate(0,nm,OBJ_RECTANGLE_LABEL,0,0,0);
   ObjectSetInteger(0,nm,OBJPROP_XDISTANCE,  x);
   ObjectSetInteger(0,nm,OBJPROP_YDISTANCE,  y);
   ObjectSetInteger(0,nm,OBJPROP_XSIZE,      w);
   ObjectSetInteger(0,nm,OBJPROP_YSIZE,      h);
   ObjectSetInteger(0,nm,OBJPROP_BGCOLOR,    C'8,8,20');
   ObjectSetInteger(0,nm,OBJPROP_BORDER_TYPE,BORDER_FLAT);
   ObjectSetInteger(0,nm,OBJPROP_COLOR,      C'70,70,120');
   ObjectSetInteger(0,nm,OBJPROP_CORNER,     CORNER_LEFT_UPPER);
   ObjectSetInteger(0,nm,OBJPROP_BACK,       false);
   ObjectSetInteger(0,nm,OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0,nm,OBJPROP_ZORDER,     5);
}

void Txt(string nm,datetime t,double p,string txt,color c,int fs)
{
   if(ObjectFind(0,nm)<0) ObjectCreate(0,nm,OBJ_TEXT,0,t,p);
   ObjectSetInteger(0,nm,OBJPROP_TIME,      t);
   ObjectSetDouble (0,nm,OBJPROP_PRICE,     p);
   ObjectSetString (0,nm,OBJPROP_TEXT,      txt);
   ObjectSetInteger(0,nm,OBJPROP_COLOR,     c);
   ObjectSetInteger(0,nm,OBJPROP_FONTSIZE,  fs);
   ObjectSetString (0,nm,OBJPROP_FONT,      "Arial Bold");
   ObjectSetInteger(0,nm,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,nm,OBJPROP_ZORDER,    9);
}
//+------------------------------------------------------------------+
