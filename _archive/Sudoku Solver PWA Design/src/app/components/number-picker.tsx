import { motion, AnimatePresence } from "motion/react";
import { X } from "lucide-react";

interface NumberPickerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (number: number) => void;
  selectedCell?: { row: number; col: number } | null;
}

export function NumberPicker({
  isOpen,
  onClose,
  onSelect,
}: NumberPickerProps) {
  const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9];

  const handleSelect = (num: number) => {
    onSelect(num);
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          />

          {/* Number Picker */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-0 left-0 right-0 z-50 bg-gray-900 rounded-t-3xl border-t border-gray-700 p-6 pb-8"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">
                Select Number
              </h3>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-full transition-colors"
              >
                <X className="size-5 text-gray-400" />
              </button>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              {numbers.map((num) => (
                <motion.button
                  key={num}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleSelect(num)}
                  className="aspect-square bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl text-white text-2xl font-bold shadow-lg hover:shadow-cyan-500/50 transition-shadow"
                >
                  {num}
                </motion.button>
              ))}
            </div>

            <button
              onClick={() => {
                onSelect(0);
                onClose();
              }}
              className="w-full py-4 bg-gray-800 rounded-2xl text-white font-medium hover:bg-gray-700 transition-colors"
            >
              Clear Cell
            </button>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
