#include "mandelbrot.h"
#include<QtGui/QPainter>
#include<QtGui/QImage>
#include<QtGui/QMouseEvent>
#include<QtCore/QTime>

void MandelbrotWidget::resizeEvent(QResizeEvent *)
{
  if(size < width() * height())
  {
    std::cout << "reallocate buffer" << std::endl;
    size = width() * height();
    if(buffer) delete[]buffer;
    buffer = new unsigned char[4*size];
  }
}

template<typename Real> int MandelbrotWidget::render(int max_iter, int img_width, int img_height)
{
  enum { packetSize = Eigen::ei_packet_traits<Real>::size }; // number of reals in a Packet
  typedef Eigen::Matrix<Real, packetSize, 1> Packet; // wrap a Packet as a vector

  int alignedWidth = (img_width/packetSize)*packetSize;
  float yradius = xradius * img_height / img_width;
  Eigen::Vector2f start(center.x() - xradius, center.y() - yradius);
  Eigen::Vector2f step(2*xradius/img_width, 2*yradius/img_height);
  int pix = 0, total_iter = 0;

  for(int y = 0; y < img_height; y++)
  {
    // for each pixel, we're going to do the iteration z := z^2 + c where z and c are complex numbers, 
    // starting with z = c = complex coord of the pixel. pzi and pzr denote the real and imaginary parts of z.
    // pci and pcr denote the real and imaginary parts of c.

    Packet pzi_start, pci_start;
    for(int i = 0; i < packetSize; i++) pzi_start[i] = pci_start[i] = start.y() + y * step.y();

    for(int x = 0; x < alignedWidth; x += packetSize, pix += packetSize)
    {
      Packet pcr, pci = pci_start, pzr, pzi = pzi_start, pzr_buf;
      for(int i = 0; i < packetSize; i++) pzr[i] = pcr[i] = start.x() + (x+i) * step.x();

      // do the iterations. Every 4 iterations we check for divergence, in which case we can stop iterating.
      int j = 0;
      typedef Eigen::Matrix<int, packetSize, 1> Packeti;
      Packeti pix_iter = Packeti::zero(), pix_dont_diverge;
      do
      {
        for(int i = 0; i < 4; i++)
        {
          pzr_buf = pzr;
          pzr = pzr.cwiseAbs2() - pzi.cwiseAbs2() + pcr;
          pzi = 2 * pzr_buf.cwiseProduct(pzi) + pci;
        }
        pix_dont_diverge = (pzr.cwiseAbs2() + pzi.cwiseAbs2())
                           .cwiseLessThan(Packet::constant(4))
                           .template cast<int>();
        pix_iter += 4 * pix_dont_diverge;
        j++;
        total_iter += 4 * packetSize;
      }
      while(j < max_iter/4 && pix_dont_diverge.any());

      // compute arbitrary pixel colors
      for(int i = 0; i < packetSize; i++)
      {
        
        buffer[4*(pix+i)] = float(pix_iter[i])*255/max_iter;
        buffer[4*(pix+i)+1] = 0;
        buffer[4*(pix+i)+2] = 0;
      }
    }

    // if the width is not a multiple of packetSize, fill the remainder in black
    for(int x = alignedWidth; x < img_width; x++, pix++)
      buffer[4*pix] = buffer[4*pix+1] = buffer[4*pix+2] = 0;
  }
  return total_iter;
}

void MandelbrotWidget::paintEvent(QPaintEvent *)
{
  float resolution  = xradius*2/width();
  int max_iter = 64;
  if(resolution < 1e-4f) max_iter += 32 * ( - 4 - std::log10(resolution));
  max_iter = (max_iter/4)*4;
  int img_width = width()/draft;
  int img_height = height()/draft;
  static float max_speed = 0;
  int total_iter;
  bool single_precision = resolution > 1e-6f;

  QTime time;
  time.start();
  if(single_precision)
    total_iter = render<float>(max_iter, img_width, img_height);
  else
    total_iter = render<double>(max_iter, img_width, img_height);
  int elapsed = time.elapsed();

  if(draft == 1)
  {
    float speed = elapsed ? float(total_iter)*1000/elapsed : 0;
    max_speed = std::max(max_speed, speed);
    std::cout << elapsed << " ms elapsed, "
              << total_iter << " iters, "
              << speed << " iters/s (max " << max_speed << ")" << std::endl;
    int packetSize = single_precision ? int(Eigen::ei_packet_traits<float>::size)
                                      : int(Eigen::ei_packet_traits<double>::size);
    setWindowTitle(QString("resolution ")+QString::number(xradius*2/width(), 'e', 2)
                  +(single_precision ? QString(", single ") : QString(", double "))
                  +QString("precision, ")
                  +(packetSize==1 ? QString("no vectorization")
                                  : QString("vectorized (%1 per packet)").arg(packetSize)));
  }
  
  QImage image(buffer, img_width, img_height, QImage::Format_RGB32);
  QPainter painter(this);
  painter.drawImage(QPoint(0, 0), image.scaled(width(), height()));

  if(draft>1)
  {
    draft /= 2;
    setWindowTitle(QString("recomputing at 1/%1 resolution...").arg(draft));
    update();
  }
}

void MandelbrotWidget::mousePressEvent(QMouseEvent *event)
{
  if( event->buttons() & Qt::LeftButton )
  {
    lastpos = event->pos();
    float yradius = xradius * height() / width();
    center = Eigen::Vector2f(center.x() + (event->pos().x() - width()/2) * xradius * 2 / width(),
                             center.y() + (event->pos().y() - height()/2) * yradius * 2 / height());
    draft = 16;
    update();
  }
}

void MandelbrotWidget::mouseMoveEvent(QMouseEvent *event)
{
  QPoint delta = event->pos() - lastpos;
  lastpos = event->pos();
  if( event->buttons() & Qt::LeftButton )
  {
    float t = 1 + 5 * float(delta.y()) / height();
    if(t < 0.5) t = 0.5;
    if(t > 2) t = 2;
    xradius *= t;
    draft = 16;
    update();
  }
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  MandelbrotWidget w;
  w.show();
  return app.exec();
}

#include "mandelbrot.moc"
