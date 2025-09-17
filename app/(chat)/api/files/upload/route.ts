// app/api/files/upload/route.ts
import { put } from '@vercel/blob';
import { NextResponse } from 'next/server';
import { z } from 'zod';
import { auth } from '@/app/(auth)/auth';

// --- Allowed types & limits ---
const ALLOWED_MIME_TYPES = new Set([
  // images (optional, keep if you still want to allow)
  'image/jpeg',
  'image/png',

  // docs/data
  'application/pdf',
  'text/csv',
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',

  // some browsers send CSV as text/plain
  'text/plain',
]);

const EXT_FALLBACKS = new Map<string, string>([
  ['.pdf', 'application/pdf'],
  ['.csv', 'text/csv'],
  ['.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
  ['.xls', 'application/vnd.ms-excel'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
]);

// 50MB for PDFs, 10MB for CSV/Excel/images
const MAX_BYTES_BY_MIME: Record<string, number> = {
  'application/pdf': 50 * 1024 * 1024,
  'text/csv': 10 * 1024 * 1024,
  'application/vnd.ms-excel': 10 * 1024 * 1024,
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
    10 * 1024 * 1024,
  'text/plain': 10 * 1024 * 1024,
  'image/jpeg': 10 * 1024 * 1024,
  'image/png': 10 * 1024 * 1024,
};

function inferMimeFromName(name: string | undefined, fallback: string) {
  if (!name) return fallback;
  const lower = name.toLowerCase();
  for (const [ext, mime] of EXT_FALLBACKS) {
    if (lower.endsWith(ext)) return mime;
  }
  return fallback;
}

const BlobSchema = z.instanceof(Blob);

export async function POST(request: Request) {
  const session = await auth();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  if (request.body === null) {
    return NextResponse.json({ error: 'Request body is empty' }, { status: 400 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get('file');
    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    // Validate it's a Blob
    const parsed = BlobSchema.safeParse(file);
    if (!parsed.success) {
      return NextResponse.json({ error: 'Invalid file payload' }, { status: 400 });
    }

    // Name (File may exist in Node 18+/Edge; use optional chaining just in case)
    const filename =
      (formData.get('file') as File | undefined)?.name || 'upload';

    // Determine content type (some CSVs come as text/plain)
    let contentType = file.type || inferMimeFromName(filename, 'application/octet-stream');

    // If browser sent text/plain but extension is .csv/.xlsx/.xls, fix it
    if (contentType === 'text/plain') {
      contentType = inferMimeFromName(filename, contentType);
    }

    // Check allowed types
    if (!ALLOWED_MIME_TYPES.has(contentType)) {
      // If type is still octet-stream, try extension one last time
      const extFixed = inferMimeFromName(filename, contentType);
      if (!ALLOWED_MIME_TYPES.has(extFixed)) {
        return NextResponse.json(
          {
            error:
              'Unsupported file type. Allowed: PDF, CSV, Excel (.xlsx, .xls), PNG, JPEG.',
          },
          { status: 400 },
        );
      }
      contentType = extFixed;
    }

    // Size limits by MIME
    const maxBytes =
      MAX_BYTES_BY_MIME[contentType] ?? 10 * 1024 * 1024; // default 10MB
    if (file.size > maxBytes) {
      const mb = Math.round(maxBytes / (1024 * 1024));
      return NextResponse.json(
        { error: `File too large. Max ${mb}MB for ${contentType}.` },
        { status: 400 },
      );
    }

    const fileBuffer = await file.arrayBuffer();

    // Upload to Vercel Blob (pass contentType so downstream clients see it)
    const data = await put(`${filename}`, fileBuffer, {
      access: 'public',
      contentType,
    });

    // Mirror the response shape your client expects
    // (url, pathname, contentType)
    return NextResponse.json({
      url: data.url,
      pathname: data.pathname,
      contentType,
    });
  } catch (err) {
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 },
    );
  }
}