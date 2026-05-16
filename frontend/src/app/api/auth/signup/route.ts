import { NextResponse } from 'next/server';

type SignUpBody = {
  name?: string;
  email?: string;
  password?: string;
};

export async function POST(request: Request) {
  const body = (await request.json()) as SignUpBody;

  const name = body.name?.trim();
  const email = body.email?.trim().toLowerCase();
  const password = body.password ?? '';

  if (!name || !email || !password) {
    return NextResponse.json({ error: 'Name, email, and password are required.' }, { status: 400 });
  }

  const user = {
    id: `demo-${Buffer.from(email).toString('base64url').slice(0, 12)}`,
    email,
    name,
  };

  const token = Buffer.from(`${user.id}:${Date.now()}`).toString('base64url');

  return NextResponse.json({ token, user }, { status: 201 });
}
