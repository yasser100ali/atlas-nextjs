import { signIn } from '@/app/(auth)/auth';
import { getToken } from 'next-auth/jwt';
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const redirectUrl = searchParams.get('redirectUrl') || '/';

  const isHttps =
    new Headers(request.headers).get('x-forwarded-proto') === 'https' ||
    new URL(request.url).protocol === 'https:';

  const token = await getToken({
    req: request,
    secret: process.env.AUTH_SECRET,
    secureCookie: isHttps,
  });

  if (token) {
    return NextResponse.redirect(new URL('/', request.url));
  }

  return signIn('guest', { redirect: true, redirectTo: redirectUrl });
}
