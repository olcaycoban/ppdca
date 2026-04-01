import { render, screen, fireEvent, within } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      structureNames: ['BrainStem', 'Chiasm'],
      patients: [
        {
          id: '0522c0555',
          scene3d: 'pddca-viz/0522c0555/scene_3d.html',
          dicePercent: { BrainStem: 85.4, Chiasm: null },
          organMetrics: {
            BrainStem: {
              dicePercent: 85.4,
              gtVoxels: 10000,
              predVoxels: 9800,
              tpVoxels: 8500,
              fnVoxels: 1500,
              fpVoxels: 1300,
            },
            Chiasm: {
              dicePercent: null,
              gtVoxels: 0,
              predVoxels: 0,
              tpVoxels: 0,
              fnVoxels: 0,
              fpVoxels: 0,
            },
          },
        },
      ],
    }),
  });
});

test('Dice tablosu ve organ tıklanınca sağda detay', async () => {
  render(<App />);
  expect(
    await screen.findByRole('heading', { name: /PDDCA — 3D karşılaştırma/i })
  ).toBeInTheDocument();
  expect(await screen.findByRole('heading', { name: /Dice \(DSC, %\)/i })).toBeInTheDocument();
  expect(screen.getByText(/Soldan bir organ adına tıklayın/i)).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /^BrainStem$/ }));

  const detailHeading = await screen.findByRole('heading', { name: /^BrainStem$/i });
  const detail = detailHeading.closest('.dice-detail');
  expect(within(detail).getByText('GT vokseller')).toBeInTheDocument();
  expect(within(detail).getByText(/Ground truth \(3D\)/i)).toBeInTheDocument();
  expect(within(detail).getByText(/TP = GT − FN =/)).toBeInTheDocument();
  expect(within(detail).getByText(/DSC% \(GT \+ tahmin paydası\)/i)).toBeInTheDocument();
});
