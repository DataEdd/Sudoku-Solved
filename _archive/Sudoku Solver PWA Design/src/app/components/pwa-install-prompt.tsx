import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Download, X } from "lucide-react";
import { GlassButton } from "./glass-button";

export function PWAInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    const handler = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e);
      
      // Check if user has dismissed before
      const dismissed = localStorage.getItem("pwa-prompt-dismissed");
      if (!dismissed) {
        setShowPrompt(true);
      }
    };

    window.addEventListener("beforeinstallprompt", handler);

    return () => {
      window.removeEventListener("beforeinstallprompt", handler);
    };
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;

    if (outcome === "accepted") {
      console.log("User accepted the install prompt");
    }

    setDeferredPrompt(null);
    setShowPrompt(false);
  };

  const handleDismiss = () => {
    setShowPrompt(false);
    localStorage.setItem("pwa-prompt-dismissed", "true");
  };

  return (
    <AnimatePresence>
      {showPrompt && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          className="fixed bottom-20 left-4 right-4 z-50 max-w-md mx-auto"
        >
          <div className="bg-gray-900/95 backdrop-blur-md border border-cyan-500/30 rounded-2xl p-4 shadow-2xl shadow-cyan-500/20">
            <div className="flex items-start gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center flex-shrink-0">
                <Download className="size-6 text-white" />
              </div>

              <div className="flex-1 min-w-0">
                <h3 className="text-white font-semibold text-sm mb-1">
                  Install App
                </h3>
                <p className="text-white/70 text-xs">
                  Add to home screen for quick access and offline use
                </p>
              </div>

              <button
                onClick={handleDismiss}
                className="p-1 hover:bg-white/10 rounded-lg transition-colors flex-shrink-0"
              >
                <X className="size-4 text-white/60" />
              </button>
            </div>

            <div className="mt-3 flex gap-2">
              <GlassButton
                variant="secondary"
                size="sm"
                onClick={handleDismiss}
                className="flex-1"
              >
                Not now
              </GlassButton>
              <GlassButton
                variant="primary"
                size="sm"
                onClick={handleInstall}
                className="flex-1"
              >
                Install
              </GlassButton>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
