import { Dialog, DialogContent } from "@/components/ui/dialog";

interface ImageZoomModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  imageSrc: string;
  altText: string;
  title?: string;
}

export function ImageZoomModal({
  isOpen,
  onOpenChange,
  imageSrc,
  altText,
}: ImageZoomModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm border-2 border-blue-200">
        <div className="flex items-center justify-center bg-slate-100 rounded-lg overflow-hidden">
          <img
            src={imageSrc}
            alt={altText}
            className="max-w-full max-h-96 object-contain"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
