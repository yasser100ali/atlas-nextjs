import type { Attachment } from '@/lib/types';
import {
  CrossSmallIcon,
  CSVIcon,
  ExcelIcon,
  PDFIcon,
  GenericFileIcon,
} from './icons';
import { Button } from './ui/button';

const COLOR_BY_TYPE: Record<string, string> = {
  'application/pdf': 'bg-red-600',
  'application/vnd.ms-excel': 'bg-green-600',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
    'bg-green-600',
  'text/csv': 'bg-blue-600',
};

const getLabel = (ct: string) => {
  if (ct === 'application/pdf') return 'PDF';
  if (ct === 'application/vnd.ms-excel') return 'XLS';
  if (
    ct === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  )
    return 'XLSX';
  if (ct === 'text/csv') return 'CSV';
  return 'FILE';
};

export const PreviewAttachment = ({
  attachment,
  isUploading = false,
  onRemove,
}: {
  attachment: Attachment;
  isUploading?: boolean;
  onRemove?: () => void;
}) => {
  const { name, contentType } = attachment;
  const color = COLOR_BY_TYPE[contentType] ?? 'bg-gray-600';

  const FileTypeIcon = () => {
    if (contentType === 'application/pdf') return <PDFIcon size={20} />;
    if (contentType === 'application/vnd.ms-excel')
      return <ExcelIcon size={20} />;
    if (
      contentType ===
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
      return <ExcelIcon size={20} />;
    if (contentType === 'text/csv') return <CSVIcon size={20} />;
    return <GenericFileIcon size={20} />;
  };

  return (
    <div
      data-testid="input-attachment-preview"
      className={`relative group inline-flex items-center gap-2 pl-2 pr-6 py-1 rounded-md text-white text-sm shadow-sm ring-1 ring-white/20 min-w-[200px] sm:min-w-[240px] ${color}`}
    >
      <div className="shrink-0">
        <FileTypeIcon />
      </div>
      <span className="text-[10px] font-semibold tracking-wide uppercase/90 hidden sm:inline">
        {getLabel(contentType)}
      </span>
      <span className="flex-1 truncate text-white/95">{name}</span>
      {onRemove && !isUploading && (
        <Button
          onClick={onRemove}
          size="icon"
          variant="ghost"
          className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full flex items-center justify-center bg-black/50 text-white hover:bg-black/70 shadow-sm ring-1 ring-black/30 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
        >
          <CrossSmallIcon size={9} />
        </Button>
      )}
    </div>
  );
};
