"use client";

import React, { useState, useEffect } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { 
  User, CreditCard, Wrench, ShieldAlert, 
  ShieldCheck, AlertTriangle, AlertCircle, FileDown, Mail 
} from "lucide-react";
import { API_URL } from "@/config";

const BLUE = "#38bdf8";
const ORG = "#f97316";
const DARK_SLATE = "#1e293b";

interface PredictionProps {
  customerData: any;
  setCustomerData: (data: any) => void;
  churnProb: number;
  setChurnProb: (prob: number) => void;
  setActiveTab: (tab: string) => void;
}


export default function Prediction({
  customerData,
  setCustomerData,
  churnProb,
  setChurnProb,
  setActiveTab
}: PredictionProps) {
  // Form state
  const [customerName, setCustomerName] = useState("");
  const [customerEmail, setCustomerEmail] = useState("");
  const [companyName, setCompanyName] = useState("");
  
  const [gender, setGender] = useState("Male");
  const [seniorCitizen, setSeniorCitizen] = useState("No");
  const [partner, setPartner] = useState("Yes");
  const [dependents, setDependents] = useState("No");
  
  const [tenure, setTenure] = useState(12);
  const [contract, setContract] = useState("Month-to-month");
  const [monthlyCharges, setMonthlyCharges] = useState(65.0);
  const [totalCharges, setTotalCharges] = useState(780.0);
  const [paperlessBilling, setPaperlessBilling] = useState("Yes");
  const [paymentMethod, setPaymentMethod] = useState("Electronic check");
  
  const [phoneService, setPhoneService] = useState("Yes");
  const [multipleLines, setMultipleLines] = useState("No");
  const [internetService, setInternetService] = useState("Fiber optic");
  const [onlineSecurity, setOnlineSecurity] = useState("No");
  const [onlineBackup, setOnlineBackup] = useState("No");
  const [deviceProtection, setDeviceProtection] = useState("No");
  const [techSupport, setTechSupport] = useState("No");
  const [streamingTV, setStreamingTV] = useState("No");
  const [streamingMovies, setStreamingMovies] = useState("No");

  // Status and response states
  const [analyzing, setAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [averages, setAverages] = useState<any>(null);
  const [emailError, setEmailError] = useState("");

  // Load averages for comparison
  useEffect(() => {
    async function loadAverages() {
      try {
        const res = await fetch(`${API_URL}/averages`);
        if (res.ok) {
          const json = await res.json();
          setAverages(json);
        }
      } catch (e) {
        console.error("Failed to load averages:", e);
      }
    }
    loadAverages();
  }, []);

  const validateEmail = (email: string) => {
    const re = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    return re.test(String(email).toLowerCase());
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateEmail(customerEmail)) {
      setEmailError("Please enter a valid email address.");
      return;
    }
    setEmailError("");
    setAnalyzing(true);

    const profileData = {
      gender,
      SeniorCitizen: seniorCitizen === "Yes" ? 1 : 0,
      Partner: partner,
      Dependents: dependents,
      tenure,
      PhoneService: phoneService,
      MultipleLines: multipleLines,
      InternetService: internetService,
      OnlineSecurity: onlineSecurity,
      OnlineBackup: onlineBackup,
      DeviceProtection: deviceProtection,
      TechSupport: techSupport,
      StreamingTV: streamingTV,
      StreamingMovies: streamingMovies,
      Contract: contract,
      PaperlessBilling: paperlessBilling,
      PaymentMethod: paymentMethod,
      MonthlyCharges: monthlyCharges,
      TotalCharges: totalCharges,
      CustomerName: customerName,
      CustomerEmail: customerEmail,
      CompanyName: companyName
    };

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profileData)
      });

      if (!res.ok) {
        throw new Error("Prediction request failed");
      }

      const json = await res.json();
      setPredictionResult(json);
      setChurnProb(json.churn_probability);
      setCustomerData(json.customer_data);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend prediction model. Verify python api server status.");
    } finally {
      setAnalyzing(false);
    }
  };

  const downloadCSV = () => {
    if (!predictionResult) return;
    const exportData = {
      ...predictionResult.customer_data,
      Prediction_Churn: predictionResult.prediction === 1 ? "Yes" : "No",
      Churn_Probability: `${predictionResult.churn_probability}%`,
      Stay_Probability: `${predictionResult.stay_probability}%`
    };

    const headers = Object.keys(exportData).join(",");
    const values = Object.values(exportData).map(val => {
      const strVal = String(val);
      return strVal.includes(",") ? `"${strVal}"` : strVal;
    }).join(",");
    const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + values;
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `churn_prediction_${(customerName.trim() || 'customer').replace(/\s+/g, '_')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Setup comparison chart data
  const comparisonData = averages && predictionResult ? [
    {
      name: "Tenure (months)",
      Customer: tenure,
      Average: Math.round(averages.tenure)
    },
    {
      name: "Monthly ($)",
      Customer: monthlyCharges,
      Average: Math.round(averages.MonthlyCharges)
    },
    {
      name: "Total (k$)",
      Customer: Math.round(totalCharges / 10) / 100, // Show in thousands
      Average: Math.round(averages.TotalCharges / 10) / 100
    }
  ] : [];

  return (
    <div className="space-y-8 animate-fade-up">
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-2">
          <Wrench className="w-6 h-6 text-[#38bdf8]" />
          <h2 className="text-xl font-bold text-white">Customer Profile Intelligence</h2>
        </div>
        <p className="text-gray-400 text-sm mb-6">
          Enter customer details below to calculate churn probability and feed details to the retention strategy engine.
        </p>

        <form onSubmit={handlePredict} className="space-y-8">
          {/* Step 1 */}
          <div className="space-y-4">
            <h3 className="text-md font-bold text-white flex items-center gap-2 border-b border-[#1e2d45] pb-2">
              <User className="w-4 h-4 text-[#38bdf8]" /> Step 1: Identity & Basics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Full Name</label>
                <input 
                  type="text" 
                  value={customerName} 
                  onChange={e => setCustomerName(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                  placeholder="e.g. John Doe"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Email Address</label>
                <input 
                  type="email" 
                  value={customerEmail} 
                  onChange={e => {
                    setCustomerEmail(e.target.value);
                    if (emailError) setEmailError("");
                  }}
                  className={`w-full bg-[#0a1628] border ${emailError ? 'border-red-500' : 'border-[#1e2d45]'} rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm`}
                  placeholder="john.doe@example.com"
                  required
                />
                {emailError && <p className="text-red-500 text-[11px] mt-1">{emailError}</p>}
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Organization</label>
                <input 
                  type="text" 
                  value={companyName} 
                  onChange={e => setCompanyName(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                  placeholder="e.g. Acme Corp"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Gender</label>
                <select 
                  value={gender} 
                  onChange={e => setGender(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Senior Citizen</label>
                <select 
                  value={seniorCitizen} 
                  onChange={e => setSeniorCitizen(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Has Partner?</label>
                <select 
                  value={partner} 
                  onChange={e => setPartner(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Has Dependents?</label>
                <select 
                  value={dependents} 
                  onChange={e => setDependents(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="space-y-4">
            <h3 className="text-md font-bold text-white flex items-center gap-2 border-b border-[#1e2d45] pb-2">
              <CreditCard className="w-4 h-4 text-[#38bdf8]" /> Step 2: Account & Subscription
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Tenure (Months)</label>
                <input 
                  type="number" 
                  min="0" max="120"
                  value={tenure} 
                  onChange={e => setTenure(Number(e.target.value))}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Monthly Charges ($)</label>
                <input 
                  type="number" 
                  step="0.01" min="0" max="300"
                  value={monthlyCharges} 
                  onChange={e => setMonthlyCharges(Number(e.target.value))}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Total Charges ($)</label>
                <input 
                  type="number" 
                  step="0.01" min="0" max="20000"
                  value={totalCharges} 
                  onChange={e => setTotalCharges(Number(e.target.value))}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Contract Type</label>
                <select 
                  value={contract} 
                  onChange={e => setContract(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Paperless Billing</label>
                <select 
                  value={paperlessBilling} 
                  onChange={e => setPaperlessBilling(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Payment Method</label>
                <select 
                  value={paymentMethod} 
                  onChange={e => setPaymentMethod(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Step 3 */}
          <div className="space-y-4">
            <h3 className="text-md font-bold text-white flex items-center gap-2 border-b border-[#1e2d45] pb-2">
              <Wrench className="w-4 h-4 text-[#38bdf8]" /> Step 3: Service & Technical Profile (Advanced)
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Phone Service</label>
                <select 
                  value={phoneService} 
                  onChange={e => setPhoneService(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Multiple Lines</label>
                <select 
                  value={multipleLines} 
                  onChange={e => setMultipleLines(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No phone service">No phone service</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Internet Service</label>
                <select 
                  value={internetService} 
                  onChange={e => setInternetService(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="DSL">DSL</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Online Security</label>
                <select 
                  value={onlineSecurity} 
                  onChange={e => setOnlineSecurity(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Online Backup</label>
                <select 
                  value={onlineBackup} 
                  onChange={e => setOnlineBackup(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Device Protection</label>
                <select 
                  value={deviceProtection} 
                  onChange={e => setDeviceProtection(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Tech Support</label>
                <select 
                  value={techSupport} 
                  onChange={e => setTechSupport(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Streaming TV</label>
                <select 
                  value={streamingTV} 
                  onChange={e => setStreamingTV(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 mb-2">Streaming Movies</label>
                <select 
                  value={streamingMovies} 
                  onChange={e => setStreamingMovies(e.target.value)}
                  className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-[#38bdf8] text-sm"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
            </div>
          </div>

          <button 
            type="submit" 
            disabled={analyzing}
            className="w-full py-4 bg-gradient-to-r from-blue-700 to-blue-800 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl font-bold tracking-wide transition shadow-lg cursor-pointer disabled:opacity-50"
          >
            {analyzing ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                Analyzing customer profile...
              </span>
            ) : "Run Churn Analysis"}
          </button>
        </form>
      </div>

      {/* Prediction Output Section */}
      {predictionResult && (
        <div className="space-y-6 animate-fade-up">
          {/* Risk Verdict Banner */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Analytical Verdict</h3>
            
            <div 
              style={{
                backgroundColor: predictionResult.prediction === 1 ? "rgba(244, 63, 94, 0.08)" : "rgba(16, 185, 129, 0.08)",
                borderColor: predictionResult.prediction === 1 ? "rgba(244, 63, 94, 0.3)" : "rgba(16, 185, 129, 0.3)"
              }}
              className="border rounded-3xl p-10 text-center flex flex-col items-center justify-center shadow-inner"
            >
              <div className="mb-4">
                {predictionResult.prediction === 1 ? (
                  <ShieldAlert className="w-16 h-16 text-red-500 filter drop-shadow-[0_0_10px_rgba(239,68,68,0.4)]" />
                ) : (
                  <ShieldCheck className="w-16 h-16 text-emerald-500 filter drop-shadow-[0_0_10px_rgba(16,185,129,0.4)]" />
                )}
              </div>
              <div 
                className={`font-extrabold text-xl tracking-widest mb-2 uppercase ${
                  predictionResult.prediction === 1 ? "text-red-500" : "text-emerald-500"
                }`}
              >
                {predictionResult.prediction === 1 ? "CRITICAL CHURN RISK" : "LOYAL CUSTOMER PROFILE"}
              </div>
              <p className="text-gray-400 text-sm mb-6">
                Customer Identity: <span className="text-white font-semibold">{customerName}</span> ({companyName})
              </p>
              <div className="flex items-baseline gap-1 mb-2">
                <span className="text-6xl font-black text-white">{predictionResult.churn_probability}</span>
                <span className={`text-2xl font-bold ${predictionResult.prediction === 1 ? "text-red-500" : "text-emerald-500"}`}>%</span>
              </div>
              <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">
                Calculated Churn Probability Score
              </div>
            </div>

            {/* Split Metrics: Gauge Chart and Probability bar */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
              {/* Score Display Card */}
              <div className="bg-[#0a1628] border border-[#1e2d45] rounded-2xl p-6 flex flex-col justify-center items-center text-center">
                <h4 className="text-sm font-semibold text-gray-400 mb-4 uppercase tracking-wider">Risk Distribution</h4>
                <div className="flex justify-between w-full max-w-md bg-[#070d1a] border border-[#1e2d45] rounded-xl p-4">
                  <div className="flex flex-col items-center">
                    <span className="text-xs text-gray-500">Retention Chance</span>
                    <span className="text-2xl font-extrabold text-emerald-400 mt-1">{predictionResult.stay_probability}%</span>
                  </div>
                  <div className="w-px bg-[#1e2d45]"></div>
                  <div className="flex flex-col items-center">
                    <span className="text-xs text-gray-500">Churn Probability</span>
                    <span className="text-2xl font-extrabold text-red-400 mt-1">{predictionResult.churn_probability}%</span>
                  </div>
                </div>
              </div>

              {/* Action Prompt */}
              <div className="bg-[#0a1628] border border-[#1e2d45] rounded-2xl p-6 flex flex-col justify-between">
                <div>
                  <h4 className="text-sm font-semibold text-white mb-2">Actionable Intelligence Ready</h4>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    This customer's ML churn classification score can be directly processed by the Strategist Agent. Click below to load the Strategist workspace.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab("AI Strategist")}
                  className="w-full mt-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-2 cursor-pointer transition"
                >
                  <Mail className="w-4 h-4" /> Open AI Strategist Workspace
                </button>
              </div>
            </div>
          </div>

          {/* Strategic Insights */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Risk Drivers */}
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                <AlertCircle className="w-5 h-5 text-red-500" /> Top Risk Drivers
              </h3>
              {predictionResult.risk_drivers.length === 0 ? (
                <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm rounded-xl p-4">
                  No critical risk drivers detected for this profile.
                </div>
              ) : (
                <div className="space-y-3">
                  {predictionResult.risk_drivers.map((driver: string, idx: number) => (
                    <div key={idx} className="bg-red-500/10 border-l-4 border-l-red-500 text-red-400 rounded p-3 text-xs leading-relaxed">
                      {driver}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Recommendations */}
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                <ShieldCheck className="w-5 h-5 text-emerald-500" /> Recommended Mitigation
              </h3>
              <div className="space-y-3">
                {predictionResult.recommendations.map((rec: string, idx: number) => (
                  <div key={idx} className="bg-emerald-500/10 border-l-4 border-l-emerald-500 text-emerald-400 rounded p-3 text-xs leading-relaxed">
                    {rec}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Customer vs Average Metrics Chart */}
          {averages && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Customer vs Average Metrics</h3>
              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
                    <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                    <YAxis stroke="#94a3b8" fontSize={12} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: "#0f1e36", borderColor: "#1e2d45" }}
                      labelStyle={{ color: "#fff" }}
                    />
                    <Legend />
                    <Bar dataKey="Customer" fill={predictionResult.prediction === 1 ? ORG : BLUE} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Average" fill={DARK_SLATE} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Export Results */}
          <div className="glass-card p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
              <h4 className="text-md font-bold text-white">Export Churn Diagnostic</h4>
              <p className="text-xs text-gray-400">Download the customer profile and ML results as a CSV spreadsheet.</p>
            </div>
            <button
              onClick={downloadCSV}
              className="inline-flex items-center gap-2 px-5 py-3 bg-[#0a1628] hover:bg-[#0f1e36] border border-[#1e2d45] hover:border-[#38bdf8] text-[#38bdf8] rounded-xl font-bold text-sm transition cursor-pointer"
            >
              <FileDown className="w-5 h-5" /> Download Prediction as CSV
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
