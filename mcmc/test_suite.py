TOP = 'top'
MID = 'mid'
LOW = 'low'


MOST_COMMON_APIS = ['java.util.ArrayList<Tau_E>.ArrayList()',
                    'java.lang.String.equals(java.lang.Object)',
                    'java.lang.Throwable.printStackTrace()',
                    'java.lang.String.length()',
                    'java.lang.Throwable.getMessage()',
                    'java.io.File.File(java.lang.String)',
                    'java.util.Arrays.asList(T[])',
                    'java.lang.String.format(java.lang.String,java.lang.Object[])',
                    'java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)',
                    'java.lang.System.currentTimeMillis()']

MID_COMMON_APIS = ['java.sql.PreparedStatement.execute()',
                   '$NOT$java.lang.String.matches(java.lang.String)',
                   'java.text.NumberFormat.format(double)',
                   'java.awt.Graphics2D.draw(java.awt.Shape)',
                   'java.util.Random.nextLong()',
                   'java.lang.StringBuilder.append(long)',
                   'java.util.concurrent.TimeUnit.toMillis(long)',
                   'java.io.File.renameTo(java.io.File)',
                   'java.io.FileInputStream.read(byte[])',
                   'java.io.DataInput.readLong()']

UNCOMMON_APIS = ['java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()',
                 'java.util.ArrayList<javax.xml.transform.Source>.add(javax.xml.transform.Source)',
                 'java.util.TreeSet<org.apache.activemq.artemis.utils.SoftValueHashMap<K,V>.AggregatedSoftReference>.add(org.apache.activemq.artemis.utils.SoftValueHashMap<K,V>.AggregatedSoftReference)',
                 'java.util.SortedMap<java.lang.String,java.nio.charset.Charset>.values()',
                 'java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)',
                 'java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)',
                 'java.util.List<com.google.logging.v2.UntypedSinkName>.size()',
                 '$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)',
                 'java.util.Map<java.lang.String,byte[]>.hashCode()',
                 '$NOT$javax.swing.JTable.isRowSelected(int)']

MOST_COMMON_DISJOINT_PAIRS = {
    'java.util.ArrayList<Tau_E>.ArrayList()': {
        'top': [
            ('java.io.OutputStream.write(byte[],int,int)', 6245),
            ('javax.xml.stream.XMLStreamWriter.getNamespaceContext()', 4505),
            ('javax.xml.stream.XMLStreamWriter.writeNamespace(java.lang.String,java.lang.String)', 4485),
            ('javax.xml.stream.XMLStreamWriter.setPrefix(java.lang.String,java.lang.String)', 4472),
            ('javax.xml.stream.XMLStreamWriter.getPrefix(java.lang.String)', 4472)],
        'mid': [
            ('java.awt.geom.AffineTransform.getShearY()', 170),
            ('java.util.concurrent.ThreadPoolExecutor.allowCoreThreadTimeOut(boolean)', 170),
            ('java.awt.geom.AffineTransform.getShearX()', 169),
            ('java.security.Signature.verify(byte[])', 167),
            ('javax.swing.JComponent.setMinimumSize(java.awt.Dimension)', 165)],
        'low': [
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1)]},
    'java.lang.String.equals(java.lang.Object)': {
        'top': [
            ('javax.xml.stream.XMLStreamWriter.setPrefix(java.lang.String,java.lang.String)', 4472),
            ('javax.xml.stream.XMLStreamWriter.getPrefix(java.lang.String)', 4472),
            ('java.nio.Buffer.position()', 4042),
            ('java.nio.Buffer.position(int)', 3213),
            ('java.awt.Frame.Frame()', 3003)],
        'mid': [
            ('java.awt.Polygon.contains(double,double)', 147),
            ('java.util.logging.Logger.setResourceBundle(java.util.logging.LogRecord)', 145),
            ('java.util.WeakHashMap<Tau_K,Tau_V>.WeakHashMap(int)', 140),
            ('java.awt.Rectangle.getLocation()', 137),
            ('java.awt.Graphics.drawPolygon(int[],int[],int)', 136)],
        'low': [
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.lang.Throwable.printStackTrace()': {
        'top': [
            ('javax.xml.stream.XMLStreamWriter.getNamespaceContext()', 4505),
            ('javax.xml.stream.XMLStreamWriter.writeNamespace(java.lang.String,java.lang.String)', 4485),
            ('javax.xml.stream.XMLStreamWriter.setPrefix(java.lang.String,java.lang.String)', 4472),
            ('javax.xml.stream.XMLStreamWriter.getPrefix(java.lang.String)', 4472),
            ('java.util.BitSet.set(int)', 3497)],
        'mid': [
            ('javax.swing.text.JTextComponent.isEditable()', 116),
            ('javax.swing.JSplitPane.getOrientation()', 114),
            ('java.awt.Component.getMinimumSize()', 111),
            ('java.util.ArrayList<java.io.File>.ArrayList<File>(int)', 107),
            ('java.util.Locale.getDisplayName(java.util.Locale)', 106)],
        'low': [
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1)]},
    'java.lang.String.length()': {
        'top': [
            ('java.lang.Thread.join()', 5963),
            ('java.util.concurrent.CountDownLatch.CountDownLatch(int)', 5775),
            ('java.util.concurrent.locks.ReentrantLock.unlock()', 5213),
            ('java.util.concurrent.locks.ReentrantLock.lock()', 5202),
            ('java.lang.Thread.setDaemon(boolean)', 4684)],
        'mid': [
            ('$NOT$java.util.logging.Logger.internalIsLoggable(java.util.logging.Level)', 154),
            ('java.io.RandomAccessFile.seek(long)', 150),
            ('javax.swing.text.AbstractDocument.putProperty(java.lang.Object,java.lang.Object)', 150),
            ('java.awt.Polygon.contains(double,double)', 147),
            ('java.util.logging.Logger.setResourceBundle(java.util.logging.LogRecord)', 145)],
        'low': [
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.lang.Throwable.getMessage()': {
        'top': [
            ('java.lang.Math.exp(double)', 4561),
            ('javax.xml.stream.XMLStreamWriter.getNamespaceContext()', 4505),
            ('javax.xml.stream.XMLStreamWriter.writeNamespace(java.lang.String,java.lang.String)', 4485),
            ('javax.xml.stream.XMLStreamWriter.setPrefix(java.lang.String,java.lang.String)', 4472),
            ('javax.xml.stream.XMLStreamWriter.getPrefix(java.lang.String)', 4472)],
        'mid': [
            ('$NOT$java.util.logging.Logger.internalIsLoggable(java.util.logging.Level)', 154),
            ('javax.swing.text.AbstractDocument.putProperty(java.lang.Object,java.lang.Object)', 150),
            ('java.awt.Polygon.contains(double,double)', 147),
            ('java.util.logging.Logger.setResourceBundle(java.util.logging.LogRecord)', 145),
            ('java.util.WeakHashMap<Tau_K,Tau_V>.WeakHashMap(int)', 140)],
        'low': [
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.io.File.File(java.lang.String)': {
        'top': [
            ('java.util.Collections.unmodifiableMap(java.util.Map)', 7350),
            ('java.lang.Number.doubleValue()', 6115),
            ('java.lang.Double.doubleValue()', 5788),
            ('java.util.concurrent.locks.ReentrantLock.unlock()', 5213),
            ('java.util.concurrent.locks.ReentrantLock.lock()', 5202)],
        'mid': [
            ('java.io.BufferedOutputStream.write(byte[],int,int)', 203),
            ('java.util.Collection<T>.iterator()', 203),
            ('java.lang.Math.signum(float)', 195),
            ('java.util.List<java.lang.Integer>.addAll(java.util.Collection)', 193),
            ('java.util.jar.Manifest.getEntries()', 192)],
        'low': [
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)',1)]},
    'java.util.Arrays.asList(T[])': {
        'top': [
            ('java.lang.Throwable.initCause(java.lang.Throwable)', 16274),
            ('java.io.ObjectInputStream.defaultReadObject()', 9730),
            ('java.lang.Math.abs(double)', 9204),
            ('java.awt.Graphics.setColor(java.awt.Color)', 6098),
            ('java.lang.Double.doubleValue()', 5788)],
        'mid': [
            ('java.util.ArrayList<java.lang.Float>.ArrayList<Float>()', 277),
            ('java.util.Map.putAll(java.util.Map)', 274), ('javax.swing.JButton.JButton(javax.swing.Action)', 267),
            ('java.awt.Component.setLocation(int,int)', 254),
            ('java.util.Collection.contains(java.lang.Object)', 254)],
        'low': [
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.lang.String.format(java.lang.String,java.lang.Object[])': {
        'top': [
            ('java.util.Collection.iterator()', 10949),
            ('java.io.ObjectInputStream.defaultReadObject()', 9730),
            ('java.util.Collections.unmodifiableSet(java.util.Set)', 7699),
            ('java.util.Iterator.next()', 7159),
            ('java.lang.Boolean.Boolean(boolean)', 4950)],
        'mid': [
            ('java.util.jar.Manifest.getEntries()', 192),
            ('java.awt.Component.getObjectLock()', 188),
            ('javax.imageio.ImageIO.createImageOutputStream(java.lang.Object)', 185),
            ('java.util.Collection<V>.size()', 182),
            ('java.awt.event.ActionEvent.ActionEvent(java.lang.Object,int,java.lang.String,long,int)', 180)],
        'low': [
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)': {
        'top': [
            ('java.lang.Boolean.valueOf(boolean)', 15121),
            ('java.util.Collection.iterator()', 10949),
            ('java.util.Locale.getDefault()', 10903),
            ('java.util.Properties.Properties()', 9322),
            ('java.lang.Boolean.valueOf(java.lang.String)', 7798)],
        'mid': [
            ('java.awt.FlowLayout.FlowLayout()', 437),
            ('javax.swing.JFileChooser.JFileChooser()', 431),
            ('java.nio.channels.SocketChannel.open()', 416),
            ('java.awt.image.BufferedImage.getGraphics()', 416),
            ('java.sql.ResultSet.getLong(java.lang.String)', 410)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)',1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.lang.System.currentTimeMillis()': {
        'top': [
            ('javax.xml.stream.XMLStreamWriter.getNamespaceContext()', 4505),
            ('javax.xml.stream.XMLStreamWriter.writeNamespace(java.lang.String,java.lang.String)', 4485),
            ('javax.xml.stream.XMLStreamWriter.setPrefix(java.lang.String,java.lang.String)', 4472),
            ('javax.xml.stream.XMLStreamWriter.getPrefix(java.lang.String)', 4472),
            ('java.security.Permission.getName()', 3586)],
        'mid': [
            ('java.awt.geom.AffineTransform.getScaleX()', 186),
            ('javax.imageio.ImageIO.createImageOutputStream(java.lang.Object)', 185),
            ('java.util.Collection<V>.size()', 182),
            ('java.security.Signature.getInstance(java.lang.String,java.security.Provider)', 179),
            ('java.text.DecimalFormat.toPattern()', 179)],
        'low': [
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('$NOT$javax.swing.JTable.isRowSelected(int)', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]}
}

MID_COMMON_DISJOINT_PAIRS = {
    'java.sql.PreparedStatement.execute()':{
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)', 93201),
            ('java.lang.String.substring(int)', 55686),
            ('java.lang.String.trim()', 51127),
            ('java.util.ArrayList<java.lang.String>.ArrayList<String>()', 50865)],
        'mid': [
            ('java.lang.StackTraceElement.getClassName()', 1101),
            ('java.net.URI.getPath()', 1094),
            ('java.lang.Class<>.isPrimitive()', 1092),
            ('java.lang.reflect.Method.getAnnotation(java.lang.Class)', 1084),
            ('java.awt.Component.getWidth()', 1076)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    '$NOT$java.lang.String.matches(java.lang.String)': {
        'top': [
            ('java.lang.System.currentTimeMillis()', 93173),
            ('java.util.HashMap<Tau_K,Tau_V>.HashMap()', 42842),
            ('java.lang.Math.max(int,int)', 40317),
            ('java.util.ArrayList<Tau_E>.ArrayList(java.util.Collection)', 38913),
            ('java.lang.Thread.start()', 38175)],
        'mid': [
            ('javax.crypto.Cipher.getInstance(java.lang.String,java.lang.String)', 984),
            ('java.util.ArrayList.iterator()', 983),
            ('java.awt.Component.getHeight()', 979),
            ('javax.security.auth.x500.X500Principal.X500Principal(java.lang.String)', 978),
            ('java.io.PrintWriter.flush()', 976)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.text.NumberFormat.format(double)': {
        'top': [
            ('java.lang.Throwable.getMessage()', 106242),
            ('java.util.Arrays.asList(T[])', 97121),
            ('java.lang.Thread.sleep(long)', 63625),
            ('java.util.HashMap<Tau_K,Tau_V>.HashMap()', 42842),
            ('java.lang.Math.max(int,int)', 40317)],
        'mid': [
            ('java.awt.event.ActionEvent.ActionEvent(java.lang.Object,int,java.lang.String)', 1037),
            ('java.awt.event.MouseEvent.getClickCount()', 1029),
            ('javax.imageio.ImageIO.write(java.awt.image.RenderedImage,java.lang.String,java.io.File)', 1022),
            ('javax.swing.tree.DefaultMutableTreeNode.getUserObject()', 1014),
            ('java.util.zip.ZipEntry.getName()', 1009)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)',1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.awt.Graphics2D.draw(java.awt.Shape)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242),
            ('java.util.Arrays.asList(T[])', 97121),
            ('java.lang.String.format(java.lang.String,java.lang.Object[])', 94816)],
        'mid': [
            ('java.awt.Color.getBlue()', 1347),
            ('java.io.IOException.IOException()', 1344),
            ('java.io.DataInput.readBoolean()', 1325),
            ('javax.swing.text.JTextComponent.getText()', 1319),
            ('javax.swing.JComponent.setOpaque(boolean)', 1314)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.Random.nextLong()': {
        'top': [
            ('java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)', 93201),
            ('java.util.Iterator.hasNext()', 63111),
            ('java.lang.Class<Tau_T>.getName()', 58150),
            ('java.lang.String.substring(int)', 55686),
            ('java.lang.String.trim()', 51127)],
        'mid': [
            ('javax.swing.JLabel.setText(java.lang.String)', 1052),
            ('java.lang.ClassLoader.getResource(java.lang.String)', 1042),
            ('java.awt.Container.getComponentCount()', 1037),
            ('java.awt.event.ActionEvent.ActionEvent(java.lang.Object,int,java.lang.String)', 1037),
            ('java.awt.event.MouseEvent.getClickCount()', 1029)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.lang.StringBuilder.append(long)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.Throwable.getMessage()', 106242),
            ('java.util.Arrays.asList(T[])', 97121),
            ('java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)', 93201),
            ('java.lang.Integer.parseInt(java.lang.String)', 75095)],
        'mid': [
            ('java.util.Calendar.set(int,int)', 1228),
            ('java.lang.reflect.Method.getParameterTypes()', 1195),
            ('java.net.DatagramSocket.isClosed()', 1193),
            ('java.util.Properties.getProperty(java.lang.String,java.lang.String)', 1191),
            ('java.lang.String.compareToIgnoreCase(java.lang.String)', 1189)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.concurrent.TimeUnit.toMillis(long)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.io.File.File(java.lang.String)', 100823),
            ('java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)', 93201),
            ('java.lang.StringBuilder.toString()', 77981),
            ('java.lang.Integer.parseInt(java.lang.String)', 75095)],
        'mid': [
            ('javax.swing.JComponent.setOpaque(boolean)', 1314),
            ('java.util.concurrent.atomic.AtomicInteger.get()', 1308),
            ('java.math.BigDecimal.valueOf(long)', 1285),
            ('java.util.regex.Matcher.group()', 1280),
            ('java.util.Vector<java.lang.String>.Vector<String>()', 1275)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.io.File.renameTo(java.io.File)': {
        'top': [
            ('java.util.Arrays.asList(T[])', 97121),
            ('java.lang.StringBuilder.toString()', 77981),
            ('java.lang.StringBuilder.StringBuilder()', 66406),
            ('java.lang.StringBuilder.append(java.lang.String)', 66128),
            ('java.util.Iterator.hasNext()', 63111)],
        'mid': [
            ('java.util.ArrayList<java.lang.Long>.ArrayList<Long>()', 1063),
            ('java.util.TreeMap<Tau_K,Tau_V>.TreeMap(java.util.Comparator)', 1053),
            ('java.net.DatagramSocket.getImpl()', 1053),
            ('javax.swing.JLabel.setText(java.lang.String)', 1052),
            ('java.lang.ClassLoader.getResource(java.lang.String)', 1042)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.io.FileInputStream.read(byte[])': {
        'top': [
            ('java.util.Iterator.hasNext()', 63111),
            ('java.util.List<java.lang.String>.add(java.lang.String)', 42899),
            ('java.util.HashMap<Tau_K,Tau_V>.HashMap()', 42842),
            ('java.lang.Math.max(int,int)', 40317),
            ('java.util.ArrayList<Tau_E>.ArrayList(java.util.Collection)', 38913)],
        'mid': [
            ('java.lang.management.ManagementFactory.getPlatformMBeanServer()', 1003),
            ('java.util.Collections.synchronizedMap(java.util.Map)', 995),
            ('java.nio.Buffer.limit(int)', 995),
            ('java.lang.Class.getConstructor(java.lang.Class[])', 990),
            ('java.io.PrintStream.print(char[])', 990)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.io.DataInput.readLong()': {
        'top': [
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.io.File.File(java.lang.String)', 100823),
            ('java.util.Arrays.asList(T[])', 97121)],
        'mid': [
            ('javax.swing.plaf.ComponentUI.installUI(javax.swing.JComponent)', 1354),
            ('java.awt.Color.getBlue()', 1347),
            ('java.io.IOException.IOException()', 1344),
            ('javax.swing.text.JTextComponent.getText()', 1319),
            ('javax.swing.JComponent.setOpaque(boolean)', 1314)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)',1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)',1)]}
}

UNCOMMON_DISJOINT_PAIRS = {
    'java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.lang.StringBuilder.append(long)', 1470),
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)',1)]},
    'java.util.ArrayList<javax.xml.transform.Source>.add(javax.xml.transform.Source)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.lang.StringBuilder.append(long)', 1470),
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.TreeSet<org.apache.activemq.artemis.utils.SoftValueHashMap<K,V>.AggregatedSoftReference>.add(org.apache.activemq.artemis.utils.SoftValueHashMap<K,V>.AggregatedSoftReference)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.util.Random.nextLong()', 1475),
            ('java.lang.StringBuilder.append(long)', 1470),
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)',1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.SortedMap<java.lang.String,java.nio.charset.Charset>.values()': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.lang.StringBuilder.append(long)', 1470),
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)',1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)': {
        'top': [
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242),
            ('java.io.File.File(java.lang.String)', 100823)],
        'mid': [
            ('java.lang.StringBuilder.append(long)', 1470),
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452)],
        'low': [
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1)]},
    'java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.util.Random.nextLong()', 1475),
            ('java.lang.StringBuilder.append(long)', 1470),
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455)],
        'low': [
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.List<com.google.logging.v2.UntypedSinkName>.size()': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452),
            ('java.io.Writer.write(char[])', 1439)],
        'low': [
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    '$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452),
            ('java.io.Writer.write(char[])', 1439),
            ('java.lang.Float.valueOf(float)', 1425)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    'java.util.Map<java.lang.String,byte[]>.hashCode()': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.lang.Throwable.getMessage()', 106242)],
        'mid': [
            ('java.util.concurrent.TimeUnit.toMillis(long)', 1465),
            ('java.io.File.renameTo(java.io.File)', 1460),
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452), ('java.io.Writer.write(char[])', 1439)],
        'low': [
            ('$NOT$java.awt.Graphics.drawImage(java.awt.Image,int,int,java.awt.image.ImageObserver)', 1),
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]},
    '$NOT$javax.swing.JTable.isRowSelected(int)': {
        'top': [
            ('java.util.ArrayList<Tau_E>.ArrayList()', 238395),
            ('java.lang.String.equals(java.lang.Object)', 201108),
            ('java.lang.Throwable.printStackTrace()', 197603),
            ('java.lang.String.length()', 172372),
            ('java.io.File.File(java.lang.String)', 100823)],
        'mid': [
            ('java.io.FileInputStream.read(byte[])', 1455),
            ('java.io.DataInput.readLong()', 1452),
            ('java.io.Writer.write(char[])', 1439),
            ('java.lang.Float.valueOf(float)', 1425),
            ('java.sql.ResultSet.getInt(java.lang.String)', 1423)],
        'low': [
            ('java.util.List<org.tvl.goworks.editor.go.formatting.FormatSpaces.Item>.add(org.tvl.goworks.editor.go.formatting.FormatSpaces.Item)', 1),
            ('java.util.Map<java.lang.String,byte[]>.hashCode()', 1),
            ('java.util.List<com.google.logging.v2.UntypedSinkName>.size()', 1),
            ('java.util.List<pythagoras.d.CrossingHelper.Edge>.remove(java.lang.Object)', 1),
            ('java.util.List<com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder>.add(com.mediaportal.ampdroid.database.SqliteAnnotationsHelper.AccessHolder)', 1)]}
}


COMPREHENSIVE_LIST = [
    ['java.util.ArrayList<Tau_E>.ArrayList()'], ['java.lang.String.equals(java.lang.Object)'], ['java.lang.Throwable.printStackTrace()'],  # most occurring apis
    ['java.sql.PreparedStatement.execute()'], ['$NOT$java.lang.String.matches(java.lang.String)'], ['java.text.NumberFormat.format(double)'],  # mid occurring apis
    ['java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()'], ['java.util.Map<java.lang.String,byte[]>.hashCode()'], ['java.util.List<com.google.logging.v2.UntypedSinkName>.size()'],  # least occurring apis
]

ONE_OF_EACH = [
    ['java.util.ArrayList<Tau_E>.ArrayList()'],  # top occurring api
    ['java.sql.PreparedStatement.execute()'],  # mid occurring api
    ['java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()'],  # least occurring api
    ['java.lang.String.length()', 'java.lang.String.substring(int,int)'],  # high-high joint pair
    ['java.lang.String.length()', 'java.io.BufferedWriter.newLine()'],  # high-mid joint pair
    ['java.lang.String.length()', 'javax.swing.JPanel.JPanel()'],  # high-low joint pair
    ['java.lang.StringBuilder.append(long)', 'java.lang.String.valueOf(long)'],  # mid-mid joint pair
    ['java.lang.StringBuilder.append(long)', 'java.lang.Thread.sleep(long)'],  # mid-low joint pair
    ['java.util.Map<java.lang.String,byte[]>.hashCode()', ],  # low-low joint pair
    ['java.lang.String.length()', 'java.lang.Thread.join()'],  # high-high disjoint pair
    ['java.lang.String.length()', 'java.io.RandomAccessFile.seek(long)'],  # high-mid disjoint pair
    ['java.lang.String.length()', '$NOT$javax.swing.JTable.isRowSelected(int)'],  # high-low disjoint pair
    ['java.lang.StringBuilder.append(long)', 'java.util.Calendar.set(int,int)'],  # mid-mid disjoint pair
    ['java.lang.StringBuilder.append(long)', 'java.util.Map<java.lang.String,byte[]>.hashCode()'],  # mid-low disjoint pair
    ['java.util.Map<java.lang.String,byte[]>.hashCode()', 'java.lang.String.String(byte[])'],  # low-low disjoint pair
]