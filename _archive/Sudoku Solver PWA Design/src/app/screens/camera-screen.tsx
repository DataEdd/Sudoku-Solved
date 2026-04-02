import { useRef } from "react";
import { Camera, Image as ImageIcon, History, Zap, ZapOff, Grid3x3 } from "lucide-react";
import { GlassButton } from "../components/glass-button";
import { motion } from "motion/react";

interface CameraScreenProps {
  onCapture: (imageUrl: string) => void;
  onGalleryClick: () => void;
  onHistoryClick: () => void;
  onManualClick: () => void;
  flashEnabled: boolean;
  onFlashToggle: () => void;
}

export function CameraScreen({
  onCapture,
  onGalleryClick,
  onHistoryClick,
  onManualClick,
  flashEnabled,
  onFlashToggle,
}: CameraScreenProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onCapture(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const triggerCamera = () => {
    cameraInputRef.current?.click();
  };

  const triggerGallery = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-hidden">
      {/* Hidden file inputs */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleCapture}
        className="hidden"
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleCapture}
        className="hidden"
      />

      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
            <Grid3x3 className="size-6 text-white" />
          </div>
          <span className="text-white font-bold text-lg">Sudoku Solver</span>
        </div>

        <GlassButton
          variant="secondary"
          size="icon"
          onClick={onFlashToggle}
        >
          {flashEnabled ? (
            <Zap className="size-5 fill-current" />
          ) : (
            <ZapOff className="size-5" />
          )}
        </GlassButton>
      </div>

      {/* Camera Viewfinder Placeholder */}
      <div className="absolute inset-0 flex items-center justify-center">
        {/* Corner Guides */}
        <div className="relative w-[80vw] h-[80vw] max-w-[400px] max-h-[400px]">
          {/* Top-left corner */}
          <motion.div
            initial={{ opacity: 0, x: -20, y: -20 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ delay: 0.2 }}
            className="absolute top-0 left-0 w-16 h-16 border-t-4 border-l-4 border-cyan-400 rounded-tl-2xl"
          />
          {/* Top-right corner */}
          <motion.div
            initial={{ opacity: 0, x: 20, y: -20 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ delay: 0.3 }}
            className="absolute top-0 right-0 w-16 h-16 border-t-4 border-r-4 border-cyan-400 rounded-tr-2xl"
          />
          {/* Bottom-left corner */}
          <motion.div
            initial={{ opacity: 0, x: -20, y: 20 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ delay: 0.4 }}
            className="absolute bottom-0 left-0 w-16 h-16 border-b-4 border-l-4 border-cyan-400 rounded-bl-2xl"
          />
          {/* Bottom-right corner */}
          <motion.div
            initial={{ opacity: 0, x: 20, y: 20 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ delay: 0.5 }}
            className="absolute bottom-0 right-0 w-16 h-16 border-b-4 border-r-4 border-cyan-400 rounded-br-2xl"
          />

          {/* Center instruction */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
              className="text-center"
            >
              <Camera className="size-12 text-white/60 mx-auto mb-2" />
              <p className="text-white/80 text-sm">
                Align puzzle within guides
              </p>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="absolute bottom-0 left-0 right-0 z-10 p-6 pb-8">
        <div className="flex items-center justify-between max-w-md mx-auto">
          {/* Gallery Button */}
          <GlassButton
            variant="secondary"
            size="icon"
            onClick={triggerGallery}
            className="w-14 h-14"
          >
            <ImageIcon className="size-6" />
          </GlassButton>

          {/* Capture Button */}
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={triggerCamera}
            className="relative w-20 h-20"
          >
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 opacity-30 blur-xl" />
            <div className="absolute inset-2 rounded-full bg-white border-4 border-cyan-400 shadow-lg" />
          </motion.button>

          {/* History Button */}
          <GlassButton
            variant="secondary"
            size="icon"
            onClick={onHistoryClick}
            className="w-14 h-14"
          >
            <History className="size-6" />
          </GlassButton>
        </div>

        {/* Manual Entry Button */}
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          onClick={onManualClick}
          className="mt-4 mx-auto block text-white/70 text-sm hover:text-white transition-colors"
        >
          Or enter puzzle manually →
        </motion.button>
      </div>
    </div>
  );
}
