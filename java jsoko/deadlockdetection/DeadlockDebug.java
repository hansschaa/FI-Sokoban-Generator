/**
 *  JSoko - A Java implementation of the game of Sokoban
 *  Copyright (c) 2012 by Matthias Meger, Germany
 *
 *  This file is part of JSoko.
 *
 *	JSoko is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package de.sokoban_online.jsoko.deadlockdetection;

import java.util.ArrayList;

import de.sokoban_online.jsoko.JSoko;
import de.sokoban_online.jsoko.PositionStorage;
import de.sokoban_online.jsoko.board.Board;
import de.sokoban_online.jsoko.board.DirectionConstants;
import de.sokoban_online.jsoko.boardpositions.AbsoluteBoardPosition;
import de.sokoban_online.jsoko.boardpositions.BoardPosition;
import de.sokoban_online.jsoko.boardpositions.IBoardPosition;
import de.sokoban_online.jsoko.boardpositions.RelativeBoardPosition;
import de.sokoban_online.jsoko.resourceHandling.Settings.SearchDirection;



/**
 * This class is for debugging, only.
 * Erst wird ein Level komplett rückwärts durchlaufen.
 * Anschließend wird das selbe Level vorwärts durchlaufen. Alle Stellungen, die vorwärts
 * als Nichtdeadlock deklariert werden, aber in der Rückwärtssuche nicht erreicht wurden,
 * sind tatsächlich Deadlocks. Auf diese Weise kann also festgestellt werden, welche Deadlock-
 * stellungen durch die Deadlockerkennung noch nicht erkannt werden.
 */
public final class DeadlockDebug extends Thread
		implements DirectionConstants
{

	// Konstanten für die Suchrichtung
	byte FORWARD_SEARCH;
	byte BACKWARD_SEARCH;

	// ArrayList, in der die erreichten Stellungen zwischenspeichert werden
	final ArrayList<IBoardPosition> boardPositionsQueueForward  = new ArrayList<IBoardPosition>(500000);
	final ArrayList<IBoardPosition> boardPositionsQueueBackward = new ArrayList<IBoardPosition>(500000);

	// Konstante für "Keine Kiste Verschoben". Dieser Wert ist der höchste Wert, der
	// in einer Stellung als Kistennummer gespeichert werden kann und kennzeichnet
	// eine Stellung als zur vorigen identisch, bis auf die Tatsache, dass einige
	// Kisten deaktiv sind (= eigentlich nicht auf dem Feld)
	final short NO_BOX_PUSHED = 511;

	// In dieser Klasse wird teilweise ein eigenens ErreichbareFelderSpielerobjekt
	// benutzt, da die Information der erreichbaren Felder über einen längeren
	// Zeitpunkt korrekt bleiben muss (also nicht durch andere Methoden überschrieben
	// werden darf)
	final Board.PlayersReachableSquares playersReachableSquares;

	// Referenz auf das Spielfeldobjekt
	final Board board;
	final int[] offset;

	// Referenz auf das Hauptobjekt
	final JSoko application;

	/** Object for storing board positions. Public for easier access. */
	private final PositionStorage positionStorage = new PositionStorage(1 << 22);

	// Wird diese Variable auf true gesetzt, so wird der Vorgangs des Lösens beendet,
	// auch wenn noch keine Lösung gefunden wurde.
	boolean isSolutionToBeAborted = false;

	/** Deadlock detection. */
	private final DeadlockDetection deadlockDetection;

	/**
	 * Creates an object for debugging the deadlock detection.
	 *
	 * @param application the reference to the main object holding all references
	 */
	public DeadlockDebug(JSoko application) {

		// Für einen bequemeren Zugriff Referenzen auf das Sokoban- und Spielfeldobjekt speichern
		this.application = application;
		board = application.board;

		// Eigenes ErreichbareFelderSpielerobjekt für diese Klasse anlegen
		playersReachableSquares = board.new PlayersReachableSquares();

		deadlockDetection = new DeadlockDetection(board);

		// Zum bequemeren Zugriff direkte Referenz speichern
		offset = board.offset;

		setDaemon(true);
	}

	// Runmethode des Threads
	@Override
	public void run() {
		showNotIdentifiedDeadlockPositions();
	}

	/**
	 * Shows all configurations, which have not been detected as deadlock
	 * during forward search.
	 */
	public void showNotIdentifiedDeadlockPositions() {

		// Die derzeitigen Kistenfelder als Zielfelder für die Rückwärtssuche setzen
		board.setGoalsBackwardsSearch();

		board.boxData.setAllBoxesNotFrozen();

		// Die Vorwärtssuche mit der aktuellen Stellung starten lassen
		IBoardPosition currentBoardPosition = new AbsoluteBoardPosition(board);

		/*
		 * Vorwärtsstellung speichern
		 */
		// Spielfeldsituation als durch die Vorwärtssuche erreicht kennzeichnen und in der
		// Hashtable speichern
		currentBoardPosition.setSearchDirection(SearchDirection.FORWARD);
		positionStorage.storeBoardPosition(currentBoardPosition);

		boardPositionsQueueForward.add(currentBoardPosition);

		/*
		 * Rückwärtsstellungen speichern
		 */

		// Alle Kisten vom Feld nehmen, um die Anfangsstellung für die Rückwärtssuche zu erstellen
		for (int boxNo = 0; boxNo < board.goalsCount; boxNo++) {
			board.removeBox(board.boxData.getBoxPosition(boxNo));
		}

		// Die erreichbaren Felder des Spielers ermitteln. Da keine Kisten auf
		// dem Feld sind, sind dies alle Felder, die überhaupt erreichbar sind.
		// Es wird das Klasseneigene Objekt genommen, da das globale Objekt im Spielfeld
		// bei der Instanziierung einer Stellung neu berrechnet = überschrieben wird.
		playersReachableSquares.update();

		// Alle Kisten auf Zielfelder stellen, um die Startposition für die Rückwärtssuche herzustellen.
		for (int position = board.firstRelevantSquare; position < board.lastRelevantSquare; position++) {
			// Falls es ein Zielfeld ist, wird die dem Zielfeld entsprechende Kiste darauf gesetzt
			if (board.isGoal(position)) {
				board.boxData.setBoxPosition(board.getGoalNo(position),
						position);
				board.setBoxWithNo(board.getGoalNo(position), position);
			}
		}

		/*
		 * Alle Stellungen, in denen alle Kisten auf Zielfeldern stehen können
		 * sofort als durch die Rückwärtssuche erreicht gekennzeichnet werden.
		 * Dazu muss der Spieler in jeden abgeschlossenen Bereich gesetzt
		 * werden. Später prüfen, ob einige der Endpositionen vielleicht
		 * unmöglich sind!
		 */

		// Nun auch noch alle Stellungen speichern, wo die Kisten auch alle auf Zielfeldern stehen,
		// der Spieler aber in anderen abgeschlossenen Bereichen steht.
		// (Nur die Felder berücksichtigen, auf denen keine Kiste steht -> isBetretbar()
		// und die zum aktiven Feld gehören (also nicht außerhalb liegen) -> g_erreichbareFelderSpieler != ...
		for (int position = board.firstRelevantSquare; position < board.lastRelevantSquare; position++) {
			if (board.isAccessible(position)
					&& playersReachableSquares.isSquareReachable(position)) {

				// Den abgeschlossenen Bereich demarkieren (Klasseneigenes Objekt, da bei der
				// Instanziierung der AbsolutenStellung das globale Feld in g_spielfeld
				// neu berechnet und damit die alten erreichbaren Felder überschrieben würden!
				playersReachableSquares.reduce(position);

				// Für die Instanziierung des Stellungobjekts ist es wichtig, dass die
				// Spielerposition korrekt gesetzt ist:
				board.playerPosition = position;

				// Die aktuelle Situation als durch die Rückwärtssuche sowohl im Rückwärtssucharray
				// als auch in der Hashtable als erreicht speichern
				currentBoardPosition = new AbsoluteBoardPosition(board);

				// Anfangsstellung als erreicht markieren
				currentBoardPosition.setSearchDirection(SearchDirection.BACKWARD);
				positionStorage.storeBoardPosition(currentBoardPosition);

				// Rückwärtssuche mit diesen Stellungen starten lassen
				boardPositionsQueueBackward.add(currentBoardPosition);
			}
		}

		// Zu Beginn stehen keine Kisten auf dem Spielfeld. Diese werden erst bei Beginn
		// der Vorwärts- bzw. Rückwärtssuche gesetzt
		for (int boxNo = 0; boxNo < board.goalsCount; boxNo++) {
			board.removeBox(board.boxData.getBoxPosition(boxNo));
		}

		// Ausgeben, dass nach Deadlockstellungen gesucht wird.
		application.applicationGUI.mainBoardDisplay.displayInfotext("Searching for not identified deadlockpositions ...");

		// Zunächst alle möglichen Stellungen durch die Rückwärtssuche ermitteln
		backwardSearch();

		// Ausgeben, dass die Rückwärtssuche beendet wurde
		application.applicationGUI.mainBoardDisplay.displayInfotext("Backwardsearch finished");

		// Die Vorwärtssuche durchführen. Alle Stellungen, die jetzt ermittelt werden und nicht
		// bereits durch die Rückwärtssuche ermittelt wurden sind Deadlockstellungen, die nicht
		// als Deadlockstellung erkannt wurden.
		forwardSearch();

		// Ausgeben, dass alle Deadlockstellungen angezeigt wurden.
		application.applicationGUI.mainBoardDisplay.displayInfotext("All deadlockpositions have been displayed.");

		application.setLevelForPlaying(application.currentLevel.getNumber());

		// Daten aus der Hashtable wieder löschen, damit der Speicher frei wird
		positionStorage.clear();

		return;
	}

	/**
	 * Erzeugt durch Verschieben alle möglichen Stellungen, die durch einen
	 * einzigen Push einer Kisten erzeugbar sind. Jede erzeugte Stellung wird in
	 * der Hashtable abgespeichert. Wurde sie bereits vorher einmal durch
	 * Verschieben erreicht, so wird sie übersprungen.
	 *
	 * Vorher wurde durch die Rückwärtssuche alle Stellungen, die keine Deadlockstellungen
	 * sind in der Hashtable gespeichert. Wird also eine Stellung erreicht, die nicht be-
	 * reits vorher durch die Rückwärtssuche erreicht wurde, so hat bei dieser Stellung
	 * die Deadlockdetection versagt.
	 * Genau diese Stellungen, bei denen die Deadlockdetection versagt werden angezeigt.
	 */
	private void forwardSearch() {

		//
		// Lokale Daten
		//
		// Nimmt eine Kistenposition auf
		int boxPosition;

		// Nimmt die mögliche neue Kistenposition auf
		int newBoxPosition = 0;

		// Stellungobjekt, dass die aktuelle Stellung aufnimmt
		BoardPosition currentBoardPosition = null;

		// Nimmt den Status einer Stellung auf (z.B. bereits durch
		// Rückwärtssuche erreicht)
		IBoardPosition oldBoardPosition;

		// Nimmt eine Stellung aus der Queue aller Stellungen auf
		BoardPosition newBoardPosition;

		// Gibt an, ob eine Stellung bereits durch die Rückwärtssuche erreicht wurde und somit
		// keine Deadlockstellung ist
		boolean boardPositionIsDeadlock = false;

		// Anzahl der gefundenen Deadlocksstellungen, die nicht als solche erkannt wurden
		int deadlockBoardPositionsCount = 0;

		//
		// Verarbeitungslogik
		//
		// Alle im letzten Zug erzeugten Stellungen durchgehen und prüfen,
		// welche Stellungen im nächsten Zug durch Schieben (= Vorwärtssuche) erzeugt werden können
		while (boardPositionsQueueForward.size() > 0) {

			// Eine neue Stellung aus der Queue holen, die als Ausgangsstellung dient
			newBoardPosition = ((BoardPosition) boardPositionsQueueForward.remove(0));

			// Das Spielfeld mit der aktuellen Stellung besetzen
			board.setBoardPosition(newBoardPosition);

			// Erreichbare Felder des Spielers ermitteln. Da diese Felder auch nach der Deadlock-
			// prüfung noch gebraucht werden, werden sie in einem eigenen Objekt berechnet.
			playersReachableSquares.update();

			// Get number of last pushed box.
			int pushedBoxNo = newBoardPosition.getBoxNo();

			// If no box has been pushed the pushed box number is set to -1, so the tunnel detection isn't performed.
			if (pushedBoxNo == NO_BOX_PUSHED) {
				pushedBoxNo = -1;
			}

			// Nun muss geprüft werden, welche Kisten in welche Richtungen verschoben werden können
			// -> welche neuen Stellungen können erzeugt werden
			for (int boxCounter = -1, boxNo; boxCounter < board.goalsCount; boxCounter++) {

				// Die aktuell zu untersuchende Kiste ist die mit der Nummer in "kistenzähler"
				boxNo = boxCounter;

				// Beim ersten Durchgang wird immer die zuletzt verschobene Kiste betrachtet,
				// damit eine Kiste möglichst in einem Stück geschoben wird.
				if (boxCounter == pushedBoxNo) {
					continue;
				}
				if (boxCounter == -1) {
					boxNo = pushedBoxNo;

					// Falls sich die verschobene Kiste in einem Tunnel befindet und nicht auf einem
					// Zielfeld steht, brauchen für diesen Push nur Verschiebungen dieser Kiste
					// geprüft werden!
					// (Es kann nur die verschobene Kiste in einem Tunnel stehen, da alle anderen
					// ja schon vorher weitergeschoben worden wären!)
					if (isBoxInTunnel(pushedBoxNo, FORWARD_SEARCH)) {
						boxCounter = board.goalsCount;
					}
				}

				// Kistenposition holen
				boxPosition = board.boxData.getBoxPosition(boxNo);

				// Verschieben in jede Richtung prüfen
				for (int direction = 0; direction < DIRS_COUNT; direction++) {
					// Mögliche neue Position der Kiste errechnen
					newBoxPosition = boxPosition + offset[direction];

					// Falls Kiste nicht in die gewünschte Richtung verschoben werden kann,
					// sofort die nächste Richtung probieren (kommt Spieler auf richtige Seite
					// und ist Zielfeld kein simple Deadlockfeld wird geprüft)
					if (!playersReachableSquares.isSquareReachable(boxPosition
                            - offset[direction])
							|| !board.isAccessibleBox(newBoxPosition)) {
						continue;
					}

					// Push durchführen und den Spieler auch auf das alte Kistenfeld setzen
					board.pushBox(boxPosition, newBoxPosition);
					board.playerPosition = boxPosition;

					// Diese Stellung gilt zunächst als Deadlockstellung.
					boardPositionIsDeadlock = true;

					// Objekt der aktuellen Stellung erzeugen (mit Referenz zur vorigen Stellung)
					// Kennzeichnen zu welcher Suchrichtung sie gehört und in welcher Iteration sie erstellt wurde
					currentBoardPosition = new RelativeBoardPosition(board, boxNo, direction, newBoardPosition);
					currentBoardPosition.setSearchDirection(SearchDirection.FORWARD);

					// Prüfen, ob diese Stellung bereits schon einmal erreicht wurde, indem versucht
					// wird sie aus dem Stellungsspeicher zu lesen
					oldBoardPosition = positionStorage.getBoardPosition(currentBoardPosition);

					// Falls die Stellung bereits vorher in der Vorwärtssuche erreicht wurde,
					// wird sofort mit der nächsten Stellung weitergemacht.
					// Wurde sie bereits vorher durch die Rückwärtssuche erreicht, handelt es sich definitiv nicht um
					// eine Deadlockstellung.
					if (oldBoardPosition != null) {

						// If the board position has been reached before in the forward search continue immediately.
						if (oldBoardPosition.getSearchDirection() == SearchDirection.FORWARD) {
							board.pushBoxUndo(newBoxPosition, boxPosition);
							continue;
						}

						// The board position has been reached during the backward search, hence it isn't a deadlock position.
						boardPositionIsDeadlock = false;
					}

					if (deadlockDetection.isDeadlock(newBoxPosition)) {
						// Push der Kiste rückgängig machen. Der Spieler wird sowieso beim nächsten
						// Aufruf wieder umgesetzt. Dies muss hier also nicht extra geschehen.
						board.pushBoxUndo(newBoxPosition, boxPosition);
						continue;
					}

					// Falls die Stellung eine Deadlockstellung ist, obwohl sie nicht als solche
					// erkannt wurde, so wird sie zur Information angezeigt (genau hierfür wurde
					// diese ganze Klasse geschrieben!)
					if (boardPositionIsDeadlock) {
						application.applicationGUI.mainBoardDisplay.displayInfotext("New deadlockposition found. ("+ (++deadlockBoardPositionsCount) + ")");
						application.redraw(true);
					}

					// Push der Kiste rückgängig machen. Der Spieler wird sowieso beim nächsten
					// Aufruf wieder umgesetzt. Dies muss hier also nicht extra geschehen.
					board.pushBoxUndo(newBoxPosition, boxPosition);

					// Stellung speichern.
					oldBoardPosition = positionStorage.storeBoardPosition(currentBoardPosition);

					// Neue Stellung durch Vorwärtssuche erreicht. Sie muss nun abgespeichert werden.
					boardPositionsQueueForward.add(currentBoardPosition);
				}
			}
		}
	}

	/**
	 * Erzeugt alle möglichen gültigen Stellungen für das aktuelle Level durch eine Rückwärtssuche
	 * und speichert sie in der Hashtable.
	 */
	private void backwardSearch() {

		// Nimmt die Kistenposition auf
		int boxPosition;

		// Nimmt die mögliche neue Kistenposition auf
		int newBoxPosition;

		// Stellungsobjekt, dass die aktuelle Stellung aufnimmt
		BoardPosition currentBoardPosition;

		// Nimmt den Status einer Stellung auf (z.B. bereits durch Vorwärtssuche erreicht)
		IBoardPosition oldBoardPosition;


		// Alle im letzten Zug erzeugten Stellungen durchgehen und prüfen, welche Stellungen im
		// nächsten Zug durch Ziehen (= Rückwärtssuche!) erzeugt werden können
		while (boardPositionsQueueBackward.size() > 0) {

			// Eine neue Stellung aus der Queue holen, die als Ausgangsstellung dient
			BoardPosition newBoardPosition = ((BoardPosition) boardPositionsQueueBackward
					.remove(0));

			// Das Spielfeld mit der aktuellen Stellung besetzen
			board.setBoardPosition(newBoardPosition);

			// Erreichbare Felder des Spielers ermitteln
			playersReachableSquares.update();

			// Get number of last pushed box.
			int pushedBoxNo = newBoardPosition.getBoxNo();

			// If no box has been pushed the pushed box number is set to -1, so the tunnel detection isn't performed.
			if (pushedBoxNo == NO_BOX_PUSHED) {
				pushedBoxNo = -1;
			}

			// Nun muss geprüft werden, welche Kisten in welche Richtungen verschoben werden können
			// -> welche neuen Stellungen können erzeugt werden
			for (int boxCounter = -1, boxNo; boxCounter < board.goalsCount; boxCounter++) {

				// Die aktuell zu untersuchende Kiste ist die mit der Nummer in "kistenzähler"
				boxNo = boxCounter;

				// Beim ersten Durchgang wird immer die zuletzt verschobene Kiste betrachtet,
				// damit eine Kiste möglichst in einem Stück geschoben wird.
				if (boxCounter == pushedBoxNo) {
					continue;
				}
				if (boxCounter == -1) {
					boxNo = pushedBoxNo;

					// Falls sich die verschobene Kiste in einem Tunnel befindet und nicht auf einem
					// Zielfeld steht, brauchen für diesen Push nur Verschiebungen dieser Kiste
					// geprüft werden!
					// (Es kann nur die verschobene Kiste in einem Tunnel stehen, da alle anderen
					// ja schon vorher weitergeschoben worden wären!)
					if (isBoxInTunnel(pushedBoxNo, BACKWARD_SEARCH)) {
						boxCounter = board.goalsCount;
					}
				}

				// Kistenposition holen
				boxPosition = board.boxData.getBoxPosition(boxNo);

				// Ziehen in jede Richtung prüfen
				for (int direction = 0; direction < DIRS_COUNT; direction++) {

					// Mögliche neue Position der Kiste errechnen
					newBoxPosition = boxPosition + offset[direction];

					// Falls Kiste nicht in die gewünschte Richtung gezogen werden kann,
					// sofort die nächste Richtung probieren. (es wird geprüft, ob der Spieler auf die
					// richtige Seite gelangen und dann einen Schritt rückwärts gehen kann und das
					// Zielfeld kein simple Deadlockfeld ist)
					if (!board.isAccessibleBox(newBoxPosition)
							|| !playersReachableSquares
                            .isSquareReachable(newBoxPosition
                                    + offset[direction])) {
						continue;
					}

					// Pull durchführen (Für die Instanziierung eines Stellungsobjekts muss
					// die Spielerposition korrekt gesetzt sein!)
					board.pushBox(boxPosition, newBoxPosition);
					board.playerPosition = newBoxPosition + offset[direction];

					// Objekt der aktuellen Stellung erzeugen (mit Referenz zur vorigen Stellung)
					// Kennzeichnen zu welcher Suchrichtung sie gehört und in welcher Iteration sie erstellt wurde
					currentBoardPosition = new RelativeBoardPosition(board, boxNo, direction, newBoardPosition);
					currentBoardPosition.setSearchDirection(SearchDirection.BACKWARD);

					// Prüfen, ob diese Stellung bereits schon einmal erreicht wurde, indem versucht
					// wird sie aus dem Stellungsspeicher zu lesen
					oldBoardPosition = positionStorage.getBoardPosition(currentBoardPosition);

					// Prüfen, ob die Stellung bereits vorher schon aufgetreten ist.
					if (oldBoardPosition != null
							&& ((BoardPosition) oldBoardPosition).getSearchDirection() == SearchDirection.BACKWARD) {
						// Push der Kiste rückgängig machen. Der Spieler wird so wieso beim nächsten
						// Aufruf wieder umgesetzt. Dies muss hier also nicht extra geschehen.
						board.pushBoxUndo(newBoxPosition, boxPosition);
						continue;
					}

					// Falls die Stellung eine Deadlockstellung ist kann sofort mit der nächsten Stellung
					// weitergemacht werden.
					if (deadlockDetection.isBackwardDeadlock(newBoxPosition)) {
						// Push der Kiste rückgängig machen. Der Spieler wird sowieso beim nächsten
						// Aufruf wieder umgesetzt. Dies muss hier also nicht extra geschehen.
						board.pushBoxUndo(newBoxPosition, boxPosition);
						continue;
					}

					// Push der Kiste rückgängig machen. Der Spieler wird so wieso beim nächsten
					// Aufruf wieder umgesetzt. Dies muss hier also nicht extra geschehen.
					board.pushBoxUndo(newBoxPosition, boxPosition);

					// Stellung speichern.
					oldBoardPosition = positionStorage.storeBoardPosition(currentBoardPosition);

					// Neue Stellung durch Rückwärtssuche erreicht. Sie muss nun abgespeichert werden.
					boardPositionsQueueBackward.add(currentBoardPosition);
				}
			}
		}
	}

	/**
	 * Returns whether vthe box with the specified box number is in a tunnel.
	 * This is important during solution search, since it implies, that the box
	 * can be moved (pushed), and all other boxes may be ignored for the current push.
	 *
	 * @param boxNo	the number of the relevant box
	 * @param searchDirection	the direction of the search
	 * @return <code>true</code> if the box is in a tunnel, and
	 * 		  <code>false</code> if the box is not in a tunnel
	 */
	private boolean isBoxInTunnel(int boxNo, byte searchDirection) {

		//
		// Daten
		//
		// Kistenposition
		int boxPosition;

		// Neue Kistenposition
		int newBoxPositon;

		//
		// Methoden
		//
		boxPosition = board.boxData.getBoxPosition(boxNo);

		// Falls die Kiste auf einem Zielfeld steht, so is ein eventueller Tunnel irrelevant.
		// Dabei muss Unterschieden werden, in welcher Richtung gesucht wird.
		if (searchDirection == FORWARD_SEARCH && board.isGoal(boxPosition)
				|| searchDirection == BACKWARD_SEARCH
				&& board.isGoalBackwardsSearch(boxPosition)) {

			return false;
		}

		/*
		 * Prüfen, ob durch Verschieben der Kiste nur Sicherheitsstellungen erzeugt werden können
		 * oder ob die Kiste überhaupt nur in eine Richtung verschoben werden kann.
		 *
		 * Sicherheitsstellung =
		 *    		   #
		 *   #$# oder  $
		 *	           #
		 */
		// Erreichbare Felder des Spielers ermitteln, wenn nur diese eine Kiste auf dem Feld ist
		board.setWall(boxPosition);
		board.playersReachableSquaresOnlyWalls.update();
		board.removeWall(boxPosition);

		// Zählt die Anzahl der möglichen Verschiebungen (bzw. Ziehungen)
		byte pushableDirectionsCount = 0;

		// Zählt die Anzahl an Sicherheitsstellungen, die entstanden sind
		byte safeSituationsCount = 0;

		// Anzahl der Richtungen, in der die Kiste tatsächlich verschoben werden kann.
		// (Dabei werden auch die anderen Kisten auf dem Spielfeld berücksichtigt)
		byte reallyPushableDirectionsCount = 0;

		// Verschieben in alle Richtungen ausprobieren
		for (byte direction = 0; direction < DIRS_COUNT; direction++) {
			if (searchDirection == FORWARD_SEARCH) {
				// Prüfen, ob Spieler auf die eine Seite kommt und die andere Seite frei ist (= keine Mauer,
				// nicht ".isBetretbar()", denn die Kisten sind ja noch auf dem Feld!)
				// Falls er es nicht kann, kann gleich in der nächsten Richtung geprüft werden.
				if (board.isWall(boxPosition + offset[direction])
						|| !board.playersReachableSquaresOnlyWalls
                        .isSquareReachable(boxPosition
                                - offset[direction])) {
					continue;
				}
			}

			//	Prüfen, ob der Spieler die Kiste in die jeweilige Richtung ziehen kann. Falls nicht,
			// gleich in der nächsten Richtung weiterprobieren.
			if (searchDirection == BACKWARD_SEARCH) {
				if (!board.playersReachableSquaresOnlyWalls
                        .isSquareReachable(boxPosition + offset[direction])
						|| !board.playersReachableSquaresOnlyWalls
                        .isSquareReachable(boxPosition + 2
                                * offset[direction])) {
					continue;
				}
			}

			// Kiste kann auf die gewünschte Position befördert werden
			pushableDirectionsCount++;

			// Prüfen, ob die Kiste auch tatsächlich verschoben werden kann (unter Berücksichtigung,
			// dass noch weitere Kisten auf dem Feld sind - die aktuell erreichbaren Felder des
			// Spielers liegen im globalen Array bereits vor)
			if (searchDirection == FORWARD_SEARCH
					&& playersReachableSquares.isSquareReachable(boxPosition
							- offset[direction])
					|| searchDirection == BACKWARD_SEARCH
					&& board.playersReachableSquaresOnlyWalls
							.isSquareReachable(boxPosition + 2
									* offset[direction])) {
				reallyPushableDirectionsCount++;
			}

			//	Position der Kiste berechnen, auf der sie stehen würde, wenn die Verschiebung
			// in die verschiebbare Richtung durchgeführt worden wäre.
			newBoxPositon = boxPosition + offset[direction];

			// Prüfen, ob eine Sicherheitsstellung erreicht wurde
			// (Unabhängig von der Verschiebungsrichtung können immer beide Achsen geprüft werden!)
			if (board.isWall(newBoxPositon + offset[UP])
					&& board.isWall(newBoxPositon + offset[DOWN])
					|| board.isWall(newBoxPositon - 1)
					&& board.isWall(newBoxPositon + 1)) {
				safeSituationsCount++;
			}
		}

		/*
		 * Wenn die Kiste nur in Sicherheitsstellungen befördert werden kann, dann ist sie
		 * quasi in einem Tunnel.
		 * Wenn die Kiste nur in eine Richtung verschoben werden kann , dann befindet sie sich
		 * ebenfalls in einem Tunnel und sie kann gleich noch einmal geschoben werden.
		 * Dies gilt allerdings nur, falls sie wirklich alle diese Sicherheitsstellungen mit dem
		 * nächsten Push erreichen kann.
		 */
		if ((pushableDirectionsCount > 0
				&& pushableDirectionsCount == safeSituationsCount || pushableDirectionsCount == 1
				&& searchDirection == FORWARD_SEARCH)
				&& pushableDirectionsCount == reallyPushableDirectionsCount) {

			return true;
		}

		// The box is not in a tunnel
		return false;
	}
}