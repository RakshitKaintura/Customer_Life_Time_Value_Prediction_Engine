import { NextResponse } from 'next/server';

type SignInBody = {
  email?: string;
  password?: string;
};

export async function POST(request: Request) {
  const body = (await request.json()) as SignInBody;

  const email = body.email?.trim().toLowerCase();
  const password = body.password ?? '';

  if (!email || !password) {
    return NextResponse.json({ error: 'Email and password are required.' }, { status: 400 });
  }

  const user = {
    id: `demo-${Buffer.from(email).toString('base64url').slice(0, 12)}`,
    email,
    name: email.split('@')[0],
  };

  const token = Buffer.from(`${user.id}:${Date.now()}`).toString('base64url');

  return NextResponse.json({ token, user }, { status: 200 });
}
