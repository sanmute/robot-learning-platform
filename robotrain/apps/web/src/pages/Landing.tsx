import { Link } from 'react-router-dom';
import NavBar from '../components/NavBar';

const FEATURES = [
  {
    icon: '⚡',
    title: '4-second training',
    desc: 'Configure objectives, click Train, and your model is ready in seconds.',
  },
  {
    icon: '🎯',
    title: 'Objective sliders',
    desc: 'Tune food collection, energy efficiency, speed, accuracy, and balance to match your use case.',
  },
  {
    icon: '📊',
    title: 'Learning curves',
    desc: 'See exactly how your robot improved during training with an interactive chart.',
  },
  {
    icon: '⬇️',
    title: 'Portable models',
    desc: 'Download your trained model as JSON and plug it into your own simulation pipeline.',
  },
];

const ROBOT_TYPES = [
  { emoji: '🏭', label: 'Warehouse', desc: 'Object retrieval & navigation' },
  { emoji: '⚙️', label: 'Manufacturing', desc: 'Precision assembly tasks' },
  { emoji: '🚀', label: 'Space', desc: 'Autonomous exploration' },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-brand-50 to-white">
      <NavBar />

      {/* Hero */}
      <section className="mx-auto max-w-4xl px-4 pt-20 pb-16 text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-brand-100 px-4 py-1.5 text-sm font-medium text-brand-700">
          🤖 Robot Learning Platform
        </div>
        <h1 className="mb-6 text-5xl font-extrabold tracking-tight text-gray-900 sm:text-6xl">
          Train your robot in{' '}
          <span className="text-brand-600">4 seconds</span>
        </h1>
        <p className="mx-auto mb-10 max-w-xl text-lg text-gray-600">
          Configure objectives, hit Train, and download a trained robot model — no GPU, no setup, no waiting.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link to="/login" className="btn-primary px-8 py-3 text-base">
            Get started free
          </Link>
          <a
            href="#how-it-works"
            className="btn-secondary px-8 py-3 text-base"
          >
            How it works
          </a>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="bg-white py-20">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="mb-12 text-center text-3xl font-bold text-gray-900">
            Three steps to a trained robot
          </h2>
          <ol className="grid gap-8 md:grid-cols-3">
            {[
              { n: '1', title: 'Configure', desc: 'Choose robot type and drag the objective sliders to match your mission.' },
              { n: '2', title: 'Train',     desc: 'Click Train. Watch the progress bar. Done in ~4 seconds.' },
              { n: '3', title: 'Deploy',    desc: 'View the learning curve, check the advantage score, download the model.' },
            ].map(({ n, title, desc }) => (
              <li key={n} className="card flex flex-col items-center text-center">
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-brand-600 text-lg font-bold text-white">
                  {n}
                </div>
                <h3 className="mb-2 text-lg font-semibold">{title}</h3>
                <p className="text-sm text-gray-500">{desc}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Robot types */}
      <section className="py-20">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="mb-12 text-center text-3xl font-bold text-gray-900">Robot environments</h2>
          <div className="grid gap-6 sm:grid-cols-3">
            {ROBOT_TYPES.map(({ emoji, label, desc }) => (
              <div key={label} className="card flex flex-col items-center gap-3 text-center">
                <span className="text-4xl">{emoji}</span>
                <h3 className="font-semibold text-gray-900">{label}</h3>
                <p className="text-sm text-gray-500">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="bg-white py-20">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="mb-12 text-center text-3xl font-bold text-gray-900">Features</h2>
          <div className="grid gap-6 sm:grid-cols-2">
            {FEATURES.map(({ icon, title, desc }) => (
              <div key={title} className="card flex gap-4">
                <span className="text-2xl">{icon}</span>
                <div>
                  <h3 className="mb-1 font-semibold text-gray-900">{title}</h3>
                  <p className="text-sm text-gray-500">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-brand-600 py-20">
        <div className="mx-auto max-w-2xl px-4 text-center">
          <h2 className="mb-4 text-3xl font-bold text-white">
            Ready to train your first robot?
          </h2>
          <p className="mb-8 text-brand-100">
            It's free. No credit card required.
          </p>
          <Link to="/login" className="inline-flex items-center gap-2 rounded-lg bg-white px-8 py-3 font-semibold text-brand-700 shadow transition hover:bg-brand-50">
            Start training →
          </Link>
        </div>
      </section>

      <footer className="py-8 text-center text-sm text-gray-400">
        © {new Date().getFullYear()} RoboTrain. Built with the Robot Learning Platform.
      </footer>
    </div>
  );
}
