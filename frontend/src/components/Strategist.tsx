"use client";

import React, { useState } from "react";
import { 
  Bot, Search, FileText, Download, Mail, 
  AlertTriangle, Play, Sparkles, CheckCircle2, ChevronRight 
} from "lucide-react";
import { API_URL } from "@/config";

// Simple mapping to render custom lucide icons in thought logs
import * as LucideIcons from "lucide-react";

interface StrategistProps {
  customerData: any;
  churnProb: number;
  setActiveTab: (tab: string) => void;
}

export default function Strategist({ customerData, churnProb, setActiveTab }: StrategistProps) {

  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [report, setReport] = useState<string | null>(null);
  const [activeProvider, setActiveProvider] = useState("");
  const [progress, setProgress] = useState(0);

  const handleStartAnalysis = async () => {
    setLoading(true);
    setReport(null);
    setLogs([]);
    setProgress(5);

    try {
      const res = await fetch(`${API_URL}/strategy/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customer_data: customerData,
          churn_prob: churnProb,
          user_query: query,
        }),
      });

      if (!res.ok || !res.body) {
        throw new Error("Strategy stream failed to start");
      }

      // ── Real-time SSE consumption via ReadableStream ──────────────────────
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let eventCount = 0;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        // Keep the last (possibly incomplete) chunk in the buffer
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;

          try {
            const payload = JSON.parse(line.slice(5).trim());

            if (payload.type === "log") {
              eventCount++;
              setLogs(prev => [...prev, payload.message]);
              // Advance progress naturally between 10% and 90%
              setProgress(Math.min(90, 10 + eventCount * 12));

            } else if (payload.type === "done") {
              setReport(payload.report);
              setActiveProvider(payload.provider);
              setProgress(100);

            } else if (payload.type === "error") {
              throw new Error(payload.message);
            }
          } catch (_) {
            // Malformed SSE chunk — skip
          }
        }
      }
    } catch (e) {
      console.error(e);
      alert("Failed to compile strategist plan. Ensure backend model is operational.");
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => {
    if (!report) return;
    const text = `SLIP — TELCO CHURN STRATEGY REPORT\n\nTarget Customer: ${customerData.CustomerName || 'Customer'}\n\n${report}`;
    const element = document.createElement("a");
    const file = new Blob([text], {type: 'text/markdown'});
    element.href = URL.createObjectURL(file);
    element.download = `retention_report_${(customerData.CustomerName || 'Customer').replace(/\s+/g, '_')}.md`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Helper to parse subject & body for mailto link
  const getMailtoLink = () => {
    if (!report) return "#";
    let subject = "Regarding your account - Important Update";
    let body = "";
    
    try {
      const emailMarker = "### 3. Draft Retention Email";
      if (report.includes(emailMarker)) {
        const emailSection = report.split(emailMarker)[1].split("### 4.")[0];
        
        if (emailSection.includes("Subject:")) {
          subject = emailSection.split("Subject:")[1].split("\n")[0].replace(/\*/g, "").trim();
        }
        
        const bodyMarker = emailSection.includes("Body:") ? "Body:" : "Subject:";
        body = emailSection.split(bodyMarker)[1] || emailSection;
        if (bodyMarker === "Subject:") {
          // Remove subject line from body
          body = body.split("\n").slice(1).join("\n").trim();
        }
      }
    } catch (e) {
      console.error("Mail parsing error:", e);
    }

    const email = customerData.CustomerEmail || "";
    return `mailto:${email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  };

  // Helper to dynamically render log icons
  const renderLogIcon = (logText: string) => {
    if (logText.includes("[lucide:brain]")) return <LucideIcons.Brain className="w-4 h-4 text-[#38bdf8] shrink-0" />;
    if (logText.includes("[lucide:search]")) return <LucideIcons.Search className="w-4 h-4 text-[#38bdf8] shrink-0" />;
    if (logText.includes("[lucide:file-text]")) return <LucideIcons.FileText className="w-4 h-4 text-[#38bdf8] shrink-0" />;
    if (logText.includes("[lucide:rocket]")) return <LucideIcons.Rocket className="w-4 h-4 text-emerald-400 shrink-0" />;
    if (logText.includes("[lucide:alert-triangle]")) return <LucideIcons.AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />;
    if (logText.includes("[lucide:lightbulb]")) return <LucideIcons.Lightbulb className="w-4 h-4 text-amber-400 shrink-0" />;
    return <ChevronRight className="w-4 h-4 text-gray-500 shrink-0" />;
  };

  const cleanLogText = (logText: string) => {
    return logText.replace(/\[lucide:[a-z0-9-]+\]\s*/g, "");
  };

  // Very simple Markdown rendering parser to style the report cleanly
  const renderReportContent = (markdownText: string) => {
    // Replace all lucide tags in the final report
    let text = markdownText.replace(/\[lucide:([a-z0-9-]+)\]/g, "");

    const lines = text.split("\n");
    return lines.map((line, idx) => {
      // Headers
      if (line.startsWith("###")) {
        return (
          <h4 key={idx} className="text-md font-bold text-[#38bdf8] mt-6 mb-3 border-b border-[#1e2d45] pb-1">
            {line.replace("###", "").trim()}
          </h4>
        );
      }
      if (line.startsWith("##")) {
        return (
          <h3 key={idx} className="text-lg font-bold text-white mt-8 mb-4">
            {line.replace("##", "").trim()}
          </h3>
        );
      }
      if (line.startsWith("#")) {
        return (
          <h2 key={idx} className="text-xl font-extrabold text-white mt-10 mb-6">
            {line.replace("#", "").trim()}
          </h2>
        );
      }

      // Blockquotes
      if (line.startsWith(">")) {
        return (
          <blockquote key={idx} className="border-l-4 border-[#38bdf8] pl-4 italic text-gray-400 my-4 py-1 bg-[#0f1e36]/30 rounded-r">
            {line.substring(1).trim()}
          </blockquote>
        );
      }

      // Lists
      if (line.trim().startsWith("*") || line.trim().startsWith("-")) {
        return (
          <li key={idx} className="text-sm text-gray-300 ml-6 list-disc mb-1.5 leading-relaxed">
            {line.trim().substring(1).trim()}
          </li>
        );
      }

      // Empty Lines
      if (line.trim() === "") {
        return <div key={idx} className="h-2" />;
      }

      // Regular Paragraph
      return (
        <p key={idx} className="text-sm text-gray-300 mb-2 leading-relaxed">
          {line}
        </p>
      );
    });
  };

  if (!customerData) {
    return (
      <div className="glass-card p-12 text-center max-w-2xl mx-auto my-12 animate-fade-up">
        <Bot className="w-16 h-16 text-[#38bdf8] mx-auto mb-4 animate-bounce" />
        <h3 className="text-xl font-bold text-white mb-2">AI Strategist Workspace Locked</h3>
        <p className="text-gray-400 text-sm mb-6">
          To run the strategist, please select or fill in a customer's profile on the Churn Prediction tab first.
        </p>
        <button
          onClick={() => setActiveTab("Churn Prediction")}
          className="px-6 py-3 bg-blue-700 hover:bg-blue-600 text-white rounded-xl font-bold text-sm transition cursor-pointer"
        >
          Go to Churn Prediction
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-up">
      <div>
        <h2 className="text-2xl font-extrabold text-white">AI-Driven Retention Strategy</h2>
        <p className="text-gray-400 text-sm mt-1">
          Utilizes LangGraph workflows and a RAG playbook repository to build strategic customer saving interventions.
        </p>
      </div>

      {/* Selected Customer & Risk Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 glass-card p-6">
          <h3 className="text-sm font-bold text-white mb-4 uppercase tracking-wider flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[#38bdf8]" /> Selected Target Profile
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
            <div>
              <span className="text-gray-500 block">Name</span>
              <span className="text-white font-semibold">{customerData.CustomerName}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Email</span>
              <span className="text-white font-semibold truncate block">{customerData.CustomerEmail}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Contract</span>
              <span className="text-white font-semibold">{customerData.Contract}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Internet Service</span>
              <span className="text-white font-semibold">{customerData.InternetService}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Monthly Charges</span>
              <span className="text-white font-semibold">${customerData.MonthlyCharges}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Tenure</span>
              <span className="text-white font-semibold">{customerData.tenure} months</span>
            </div>
          </div>
        </div>

        <div className="bg-[#0f1e36] border border-[#1e2d45] rounded-2xl p-6 flex flex-col justify-center items-center text-center">
          <span className="text-gray-400 text-xs font-bold uppercase tracking-wider">Calculated Risk Score</span>
          <span className="text-4xl font-black text-white mt-2 mb-1">{churnProb.toFixed(1)}%</span>
          <span className={`px-3 py-1 rounded-full text-xs font-bold ${
            churnProb > 50 
              ? "bg-red-500/10 text-red-400 border border-red-500/20" 
              : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
          }`}>
            {churnProb > 50 ? "CRITICAL PRIORITY" : "MAINTENANCE MODE"}
          </span>
        </div>
      </div>

      {/* Query Directives */}
      <div className="glass-card p-6">
        <h3 className="text-md font-bold text-white mb-2">🔍 Specific Retention Directives</h3>
        <p className="text-xs text-gray-400 mb-4">
          Provide custom constraints to target specific retention goals (e.g. price sensitivity, family bundle focus, etc.)
        </p>
        <textarea
          rows={3}
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g. Propose a long-term contract with a focus on price sensitivity, or suggest a bundle for their children."
          className="w-full bg-[#0a1628] border border-[#1e2d45] rounded-xl px-4 py-3 text-white focus:outline-none focus:border-[#38bdf8] text-sm mb-4 placeholder-gray-500"
        />
        <button
          onClick={handleStartAnalysis}
          disabled={loading}
          className="w-full py-3 bg-gradient-to-r from-blue-700 to-blue-800 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl font-bold tracking-wide transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          <Play className="w-4 h-4 fill-white" /> Start Expert AI Analysis
        </button>
      </div>

      {/* Execution logs */}
      {(loading || logs.length > 0) && (
        <div className="glass-card p-6 space-y-4">
          <div className="flex justify-between items-center text-sm">
            <span className="text-white font-semibold">Strategist Processing Logs</span>
            <span className="text-gray-400 text-xs font-mono">{progress}%</span>
          </div>
          {/* Progress Bar */}
          <div className="w-full bg-[#070d1a] h-1.5 rounded-full overflow-hidden border border-[#1e2d45]">
            <div 
              style={{ width: `${progress}%` }} 
              className="bg-blue-500 h-full rounded-full transition-all duration-300"
            ></div>
          </div>
          {/* logs container */}
          <div className="bg-[#070d1a] border border-[#1e2d45] rounded-xl p-4 space-y-2 max-h-[200px] overflow-y-auto font-mono text-xs text-gray-400">
            {logs.map((log, idx) => (
              <div key={idx} className="flex items-start gap-2.5 leading-relaxed py-0.5 animate-fade-up">
                {renderLogIcon(log)}
                <span>{cleanLogText(log)}</span>
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-[#38bdf8] animate-pulse">
                <span className="w-2.5 h-2.5 border border-t-transparent border-[#38bdf8] rounded-full animate-spin"></span>
                <span>Agent node processing...</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Report results */}
      {report && (
        <div className="space-y-6 animate-fade-up">
          {/* Main report shell */}
          <div className="glass-card p-8">
            <div className="flex justify-between items-start border-b border-[#1e2d45] pb-4 mb-6">
              <div>
                <h3 className="text-lg font-extrabold text-white">Strategic Retention Plan</h3>
                <p className="text-xs text-gray-400 mt-1">Compiled via **{activeProvider}**</p>
              </div>
              <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" /> Plan Ready
              </div>
            </div>
            
            <div className="space-y-4 text-gray-300">
              {renderReportContent(report)}
            </div>
          </div>

          {/* Action buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <button
              onClick={downloadReport}
              className="flex items-center justify-center gap-2 py-4 bg-[#0a1628] hover:bg-[#0f1e36] border border-[#1e2d45] hover:border-[#38bdf8] text-[#38bdf8] rounded-xl font-bold text-sm transition cursor-pointer"
            >
              <Download className="w-5 h-5" /> Download Full Retention Report
            </button>
            <a
              href={getMailtoLink()}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 py-4 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 border border-emerald-500/20 text-white rounded-xl font-bold text-sm transition cursor-pointer shadow-lg shadow-emerald-900/20"
            >
              <Mail className="w-5 h-5" /> Send Action Plan via Email
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
